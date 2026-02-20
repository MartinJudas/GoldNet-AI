#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  PHOENIX v5 EVOLUTION — Self-Improving Gold Futures Engine                  ║
║                                                                             ║
║  Modes:                                                                     ║
║    • EVOLVE — Continuously mutates hyperparams, prunes features,            ║
║      breeds best configs, tracks experiments on a leaderboard.              ║
║      Runs unattended for hours. Leave it at school, come back smarter.      ║
║    • SINGLE — One-shot pipeline run (original Lane A behavior).             ║
║                                                                             ║
║  Dashboard: Real-time charts, equity curves, experiment comparisons,        ║
║  heatmaps, animated feature importance, live fold stream.                   ║
║                                                                             ║
║  VM: c2d-standard-32 · us-central1-f                                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, json, logging, threading, traceback, warnings, copy, random, hashlib
from datetime import datetime, timezone, timedelta
from collections import deque, OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhoenixConfig:
    target_ticker: str = "GC=F"
    lookback_years: int = 15

    tickers: Dict[str, str] = field(default_factory=lambda: {
        "gold":"GC=F","silver":"SI=F","copper":"HG=F","oil":"CL=F",
        "tnx":"^TNX","tyx":"^TYX","irx":"^IRX","dxy":"DX-Y.NYB",
        "eurusd":"EURUSD=X","usdjpy":"JPY=X","vix":"^VIX","sp500":"^GSPC",
        "nasdaq":"^IXIC","tips":"TIP","hyg":"HYG","gld_etf":"GLD","gdx":"GDX",
    })

    train_window_min: int = 252
    test_window: int = 5
    expanding: bool = True

    model_type: str = "gbt"
    gbt_max_depth: int = 4
    gbt_n_estimators: int = 200
    gbt_learning_rate: float = 0.05
    gbt_subsample: float = 0.8
    gbt_min_samples_leaf: int = 20

    rf_n_estimators: int = 150
    rf_max_depth: int = 6

    slippage_bps: float = 5.0
    commission_bps: float = 2.0
    execution_delay_days: int = 1
    vol_target: float = 0.15
    vol_lookback: int = 21
    max_leverage: float = 2.0

    gate_raw_sharpe: float = 0.40
    gate_deflated_sharpe: float = 0.50
    gate_max_fold_failure: float = 0.50

    flask_host: str = "0.0.0.0"
    flask_port: int = 5000

    max_generations: int = 9999
    population_size: int = 6
    elite_keep: int = 2
    mutation_rate: float = 0.3


CFG = PhoenixConfig()


# ═══════════════════════════════════════════════════════════════════════════════
#  EVENT BUS
# ═══════════════════════════════════════════════════════════════════════════════

class EventBus:
    def __init__(self, maxlen=3000):
        self.logs = deque(maxlen=maxlen)
        self.signals = deque(maxlen=500)
        self.metrics = deque(maxlen=300)
        self.experiments = []
        self.leaderboard = []
        self.current_gen = 0
        self.total_folds_run = 0
        self.best_sharpe_ever = -999
        self.evolution_state = {
            "status": "idle", "generation": 0, "experiment": 0,
            "current_step": "", "progress": 0,
            "fold_results": [], "gate_result": None,
            "feature_importance": {}, "sign_audit": {},
            "equity_curve": [], "sharpe_history": [],
            "config_label": "", "config_details": {},
        }
        self._sio = None
        self._lock = threading.Lock()

    def bind(self, sio): self._sio = sio

    def log(self, msg, level="info", source="phoenix"):
        e = {"ts": datetime.now(timezone.utc).isoformat(), "level": level, "source": source, "msg": msg}
        with self._lock: self.logs.append(e)
        if self._sio: self._sio.emit("log", e)

    def signal(self, data):
        data["ts"] = datetime.now(timezone.utc).isoformat()
        with self._lock: self.signals.append(data)
        if self._sio: self._sio.emit("signal", data)

    def metric(self, data):
        data["ts"] = datetime.now(timezone.utc).isoformat()
        with self._lock: self.metrics.append(data)
        if self._sio: self._sio.emit("sys_metrics", data)

    def update(self, **kw):
        with self._lock: self.evolution_state.update(kw)
        if self._sio: self._sio.emit("evo_state", self.evolution_state)

    def add_experiment(self, exp):
        with self._lock:
            self.experiments.append(exp)
            self.leaderboard = sorted(self.experiments, key=lambda x: x.get("score", -999), reverse=True)[:15]
            if exp.get("score", -999) > self.best_sharpe_ever:
                self.best_sharpe_ever = exp["score"]
        if self._sio: self._sio.emit("experiment", exp)
        if self._sio: self._sio.emit("leaderboard", self.leaderboard)

    def fold_tick(self, fold_data):
        self.total_folds_run += 1
        if self._sio: self._sio.emit("fold_tick", {**fold_data, "total_folds": self.total_folds_run})


BUS = EventBus()

class BusHandler(logging.Handler):
    def emit(self, record):
        lvl = {10:"debug",20:"info",30:"warn",40:"error",50:"error"}.get(record.levelno,"info")
        BUS.log(self.format(record), level=lvl, source=record.name)

LOG = logging.getLogger("phoenix")
LOG.setLevel(logging.DEBUG)
LOG.addHandler(BusHandler())
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-7s %(message)s"))
LOG.addHandler(_sh)


# ═══════════════════════════════════════════════════════════════════════════════
#  THESIS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlphaThesis:
    name: str; bucket: str; description: str; expected_sign: int; confidence: str = "high"

THESIS = OrderedDict()
def reg(n, b, d, s, c="high"): THESIS[n] = AlphaThesis(n, b, d, s, c)

reg("real_yield_10y","rates","Higher real yields bearish gold",-1)
reg("real_yield_chg_5d","rates","Rising real yields bearish",-1)
reg("yield_curve_10y2y","rates","Steepening curve risk-on bearish gold",-1)
reg("rate_momentum_21d","rates","Sustained rate rises compress gold",-1)
reg("tips_return_21d","rates","TIPS rally bullish gold",+1)
reg("dxy_level","fx","Strong dollar bearish gold",-1)
reg("dxy_return_5d","fx","Dollar momentum bearish gold",-1)
reg("dxy_return_21d","fx","Sustained dollar strength bearish",-1)
reg("eurusd_return_5d","fx","Euro strength bullish gold",+1)
reg("usdjpy_return_5d","fx","Yen strength bullish gold",-1)
reg("vix_level","vol","High VIX flight to gold",+1)
reg("vix_chg_5d","vol","Rising VIX gold bid",+1)
reg("vix_term_structure","vol","Inverted VIX stress gold bid",-1)
reg("sp500_return_21d","vol","Equity weakness rotation to gold",-1)
reg("credit_spread_chg","vol","Widening spreads risk-off gold bid",-1)
reg("silver_gold_ratio","commodities","Rising silver/gold risk-on",-1)
reg("copper_gold_ratio","commodities","Rising copper/gold growth bearish gold",-1)
reg("oil_return_21d","commodities","Oil rally inflation bullish gold",+1)
reg("gold_silver_mom","commodities","Gold outperform silver continuation",+1)
reg("gold_equity_corr_63d","correlation","Negative gold-equity corr hedge signal",-1)
reg("gold_usd_corr_63d","correlation","Strong negative gold-USD normal regime",-1)
reg("gdx_gold_ratio","correlation","Miners leading gold bullish",+1)
reg("gdx_return_5d","correlation","Miner momentum confirms gold",+1)
reg("gold_mom_5d","structure","Short-term momentum",+1)
reg("gold_mom_21d","structure","Medium-term momentum",+1)
reg("gold_mom_63d","structure","Quarterly momentum",+1)
reg("gold_vol_21d","structure","High realized vol slightly bullish",+1)
reg("gold_vol_ratio","structure","Vol expansion regime change",+1,"medium")
reg("gold_rsi_14","structure","Extreme RSI mean-reversion",-1,"medium")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════════

_DATA_CACHE = {}

def fetch_data(cfg):
    global _DATA_CACHE
    if _DATA_CACHE:
        LOG.info("Using cached market data")
        return _DATA_CACHE
    LOG.info("Fetching market data …")
    BUS.update(current_step="data_fetch", progress=5)
    end = datetime.now()
    start = end - timedelta(days=cfg.lookback_years * 365)
    data = {}
    for i, (name, ticker) in enumerate(cfg.tickers.items()):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[name] = df
            LOG.info(f"  ✓ {name} ({ticker}) — {len(df)} rows")
        except Exception as e:
            LOG.error(f"  ✗ {name} — {e}")
    _DATA_CACHE = data
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def safe_ret(s, p): return s.pct_change(p)
def safe_ratio(a, b): return a / b.replace(0, np.nan)
def roll_corr(a, b, w): return a.rolling(w).corr(b)
def rsi(s, p=14):
    d = s.diff(); g = d.where(d>0,0.).rolling(p).mean(); l = (-d.where(d<0,0.)).rolling(p).mean()
    return 100 - 100/(1 + g/l.replace(0, np.nan))

def build_features(data, cfg):
    BUS.update(current_step="features", progress=15)
    closes = pd.DataFrame()
    for name, df in data.items():
        if "Close" in df.columns: closes[name] = df["Close"]
    closes = closes.sort_index().ffill().dropna(how="all")
    if "gold" not in closes.columns: raise ValueError("No gold data")
    gold = closes["gold"]
    f = pd.DataFrame(index=closes.index)
    if "tnx" in closes and "tips" in closes:
        f["real_yield_10y"] = closes["tnx"] - safe_ret(closes["tips"],252).rolling(21).mean()*100
        f["real_yield_chg_5d"] = f["real_yield_10y"].diff(5)
    if "tnx" in closes and "irx" in closes: f["yield_curve_10y2y"] = closes["tnx"] - closes["irx"]
    if "tnx" in closes: f["rate_momentum_21d"] = closes["tnx"].diff(21)
    if "tips" in closes: f["tips_return_21d"] = safe_ret(closes["tips"],21)
    if "dxy" in closes:
        f["dxy_level"] = closes["dxy"]; f["dxy_return_5d"] = safe_ret(closes["dxy"],5)
        f["dxy_return_21d"] = safe_ret(closes["dxy"],21)
    if "eurusd" in closes: f["eurusd_return_5d"] = safe_ret(closes["eurusd"],5)
    if "usdjpy" in closes: f["usdjpy_return_5d"] = safe_ret(closes["usdjpy"],5)
    if "vix" in closes:
        f["vix_level"] = closes["vix"]; f["vix_chg_5d"] = closes["vix"].diff(5)
        f["vix_term_structure"] = closes["vix"]/closes["vix"].rolling(21).mean()-1
    if "sp500" in closes: f["sp500_return_21d"] = safe_ret(closes["sp500"],21)
    if "hyg" in closes: f["credit_spread_chg"] = -safe_ret(closes["hyg"],5)
    if "silver" in closes:
        f["silver_gold_ratio"] = safe_ratio(closes["silver"],gold)
        f["gold_silver_mom"] = safe_ret(gold,21) - safe_ret(closes["silver"],21)
    if "copper" in closes: f["copper_gold_ratio"] = safe_ratio(closes["copper"],gold)
    if "oil" in closes: f["oil_return_21d"] = safe_ret(closes["oil"],21)
    gr = gold.pct_change()
    if "sp500" in closes: f["gold_equity_corr_63d"] = roll_corr(gr, closes["sp500"].pct_change(), 63)
    if "dxy" in closes: f["gold_usd_corr_63d"] = roll_corr(gr, closes["dxy"].pct_change(), 63)
    if "gdx" in closes:
        f["gdx_gold_ratio"] = safe_ratio(closes["gdx"],gold); f["gdx_return_5d"] = safe_ret(closes["gdx"],5)
    f["gold_mom_5d"] = safe_ret(gold,5); f["gold_mom_21d"] = safe_ret(gold,21)
    f["gold_mom_63d"] = safe_ret(gold,63)
    f["gold_vol_21d"] = gr.rolling(21).std()*np.sqrt(252)
    f["gold_vol_ratio"] = gr.rolling(10).std()/gr.rolling(63).std().replace(0,np.nan)
    f["gold_rsi_14"] = rsi(gold,14)
    f = f[[c for c in f.columns if c in THESIS]]
    return f, gold

def build_target(gold, delay=1):
    fwd = gold.pct_change().shift(-delay)
    return (fwd > 0).astype(int).rename("target")


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGN AUDIT
# ═══════════════════════════════════════════════════════════════════════════════

def sign_audit(features, gold_returns):
    results = {}
    for fname in features.columns:
        if fname not in THESIS: continue
        t = THESIS[fname]; corr = features[fname].corr(gold_returns)
        rsign = +1 if corr > 0 else -1
        results[fname] = {"bucket":t.bucket,"expected":t.expected_sign,
                          "corr":round(corr,4),"sign":rsign,"match":rsign==t.expected_sign}
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sharpe(r, ann=252):
    if r.std()==0 or len(r)<2: return 0.0
    return float(r.mean()/r.std()*np.sqrt(ann))

def max_dd(eq):
    pk = eq.expanding().max(); dd = (eq-pk)/pk; return float(dd.min())

def make_model(cfg):
    if cfg.model_type == "rf":
        return RandomForestClassifier(n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
                                      min_samples_leaf=cfg.gbt_min_samples_leaf, random_state=42, n_jobs=-1)
    elif cfg.model_type == "logistic":
        return LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    else:
        return GradientBoostingClassifier(max_depth=cfg.gbt_max_depth, n_estimators=cfg.gbt_n_estimators,
                                          learning_rate=cfg.gbt_learning_rate, subsample=cfg.gbt_subsample,
                                          min_samples_leaf=cfg.gbt_min_samples_leaf, random_state=42)

def walk_forward(features, target, gold, cfg, feature_subset=None, emit_folds=True):
    if feature_subset:
        features = features[[f for f in feature_subset if f in features.columns]]
    common = features.dropna().index.intersection(target.dropna().index)
    X, y = features.loc[common], target.loc[common]
    gold_a = gold.loc[common]; gr = gold_a.pct_change().fillna(0)
    n = len(X); cost = (cfg.slippage_bps + cfg.commission_bps)/10000
    folds = []; equity_points = [1.0]; cumret = 1.0
    fold_id = 0; model = None
    while True:
        te_s = cfg.train_window_min + fold_id * cfg.test_window
        te_e = te_s + cfg.test_window
        if te_e > n: break
        fold_id += 1
        Xtr, ytr = X.iloc[:te_s], y.iloc[:te_s]
        Xte, yte = X.iloc[te_s:te_e], y.iloc[te_s:te_e]
        grt = gr.iloc[te_s:te_e]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr.fillna(0))
        Xte_s = scaler.transform(Xte.fillna(0))
        model = make_model(cfg)
        model.fit(Xtr_s, ytr)
        preds = model.predict(Xte_s)
        acc = accuracy_score(yte, preds)
        pos = pd.Series(np.where(preds==1,1.,-1.), index=grt.index)
        rvol = gr.iloc[max(0,te_s-cfg.vol_lookback):te_s].std()*np.sqrt(252)
        vs = min(cfg.vol_target/rvol, cfg.max_leverage) if rvol > 0 else 1.0
        pos *= vs
        sr = pos * grt
        costs = pos.diff().abs().fillna(0)*cost
        sr_net = sr - costs
        sharpe = compute_sharpe(sr_net, 252)
        eq = (1+sr_net).cumprod()
        mdd = max_dd(eq)
        ret_net = float(sr_net.sum())
        cumret *= (1 + ret_net)
        equity_points.append(cumret)
        fd = {
            "fold_id":fold_id, "test_start":str(Xte.index[0].date()), "test_end":str(Xte.index[-1].date()),
            "train_size":len(Xtr), "test_size":len(Xte), "accuracy":round(acc,4),
            "precision":round(precision_score(yte,preds,zero_division=0),4),
            "recall":round(recall_score(yte,preds,zero_division=0),4),
            "f1":round(f1_score(yte,preds,zero_division=0),4),
            "return":round(ret_net,6), "sharpe":round(sharpe,4), "mdd":round(mdd,4),
            "long_pct":round(float((preds==1).mean()),4), "passed":sharpe>0,
            "cumulative_return":round(cumret,4),
        }
        folds.append(fd)
        if emit_folds:
            BUS.fold_tick(fd)
            if fold_id % 20 == 0:
                BUS.update(fold_results=folds[-50:], equity_curve=equity_points[-200:],
                           progress=min(90, int(te_e/n*90)))
    fi = {}
    if model and hasattr(model, 'feature_importances_'):
        fi = dict(sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True))
    return folds, equity_points, fi


# ═══════════════════════════════════════════════════════════════════════════════
#  DEFLATED SHARPE + GATE
# ═══════════════════════════════════════════════════════════════════════════════

def deflated_sharpe(sharpes, T=5):
    if len(sharpes)<2: return 0.0
    s = np.array(sharpes); mu, std = s.mean(), s.std(ddof=1)
    N = len(s); em = 0.5772156649
    e_max = std*((1-em)*sp_stats.norm.ppf(1-1/N)+em*sp_stats.norm.ppf(1-1/(N*np.e)))
    sk, ku = sp_stats.skew(s), sp_stats.kurtosis(s, fisher=True)
    num = (mu - e_max)*np.sqrt(T-1)
    den = np.sqrt(1 - sk*mu + (ku-1)/4*mu**2)
    if den == 0: return 0.0
    return float(sp_stats.norm.cdf(num/den))

def validation_gate(folds, cfg):
    if not folds: return {"passed":False,"reason":"no folds","score":-999}
    sharpes = [f["sharpe"] for f in folds]
    returns = [f["return"] for f in folds]
    failures = sum(1 for f in folds if not f["passed"])
    fr = failures/len(folds)
    rm, rs = np.mean(returns), np.std(returns, ddof=1) if len(returns)>1 else 1e-10
    raw_s = rm/rs*np.sqrt(252/cfg.test_window)
    dsr = deflated_sharpe(sharpes, T=cfg.test_window)
    cs = raw_s >= cfg.gate_raw_sharpe
    cd = dsr >= cfg.gate_deflated_sharpe
    cf = fr < cfg.gate_max_fold_failure
    score = raw_s * 0.4 + dsr * 0.3 + (1 - fr) * 0.3
    return {
        "passed":cs and cd and cf, "raw_sharpe":round(raw_s,4), "deflated_sharpe":round(dsr,4),
        "fold_failure_rate":round(fr,4), "check_sharpe":cs, "check_dsr":cd, "check_folds":cf,
        "num_folds":len(folds), "num_failures":failures,
        "avg_return":round(float(np.mean(returns)),6), "avg_sharpe":round(float(np.mean(sharpes)),4),
        "best_sharpe":round(float(max(sharpes)),4), "worst_sharpe":round(float(min(sharpes)),4),
        "total_return":round(float(sum(returns)),4), "score":round(score,4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  EVOLUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

ALL_FEATURES = list(THESIS.keys())

def random_config(base=None):
    c = copy.deepcopy(base or CFG)
    c.model_type = random.choice(["gbt","gbt","gbt","rf","logistic"])
    if c.model_type == "gbt":
        c.gbt_max_depth = random.choice([2,3,4,5,6])
        c.gbt_n_estimators = random.choice([50,100,150,200,300,400])
        c.gbt_learning_rate = random.choice([0.01,0.02,0.05,0.08,0.1,0.15])
        c.gbt_subsample = random.choice([0.5,0.6,0.7,0.8,0.9,1.0])
        c.gbt_min_samples_leaf = random.choice([5,10,15,20,30,50])
    elif c.model_type == "rf":
        c.rf_n_estimators = random.choice([50,100,150,200,300])
        c.rf_max_depth = random.choice([3,4,5,6,8,10,None])
    c.vol_target = random.choice([0.10,0.12,0.15,0.18,0.20])
    c.max_leverage = random.choice([1.5,2.0,2.5,3.0])
    c.execution_delay_days = random.choice([1,1,1,2])
    c.slippage_bps = random.choice([3,5,7,10])
    return c

def random_feature_subset():
    n = random.randint(8, len(ALL_FEATURES))
    return sorted(random.sample(ALL_FEATURES, n))

def mutate_config(parent):
    c = copy.deepcopy(parent)
    if random.random() < 0.3: c.model_type = random.choice(["gbt","rf","logistic"])
    if c.model_type == "gbt":
        if random.random() < 0.4: c.gbt_max_depth = max(2, c.gbt_max_depth + random.choice([-1,0,0,1]))
        if random.random() < 0.4: c.gbt_n_estimators = max(50, c.gbt_n_estimators + random.choice([-50,0,50,100]))
        if random.random() < 0.3: c.gbt_learning_rate = float(np.clip(c.gbt_learning_rate * random.choice([0.5,0.8,1.0,1.2,2.0]), 0.005, 0.3))
        if random.random() < 0.3: c.gbt_subsample = float(np.clip(c.gbt_subsample + random.choice([-0.1,0,0.1]), 0.4, 1.0))
    if random.random() < 0.2: c.vol_target = float(np.clip(c.vol_target + random.choice([-0.03,0,0.03]), 0.05, 0.30))
    if random.random() < 0.2: c.max_leverage = random.choice([1.5,2.0,2.5,3.0])
    return c

def config_label(cfg, feats=None):
    parts = [cfg.model_type.upper()]
    if cfg.model_type == "gbt": parts += [f"d{cfg.gbt_max_depth}",f"n{cfg.gbt_n_estimators}",f"lr{cfg.gbt_learning_rate}"]
    elif cfg.model_type == "rf": parts += [f"d{cfg.rf_max_depth}",f"n{cfg.rf_n_estimators}"]
    parts.append(f"vol{cfg.vol_target}")
    if feats: parts.append(f"f{len(feats)}")
    return " · ".join(parts)

def config_details(cfg, feats=None):
    d = {"model":cfg.model_type,"vol_target":cfg.vol_target,"max_lev":cfg.max_leverage,
         "slip_bps":cfg.slippage_bps,"delay":cfg.execution_delay_days}
    if cfg.model_type == "gbt":
        d.update({"depth":cfg.gbt_max_depth,"n_est":cfg.gbt_n_estimators,
                  "lr":cfg.gbt_learning_rate,"subsample":cfg.gbt_subsample,"min_leaf":cfg.gbt_min_samples_leaf})
    elif cfg.model_type == "rf":
        d.update({"depth":cfg.rf_max_depth,"n_est":cfg.rf_n_estimators})
    if feats: d["features"] = feats
    return d


def evolution_loop(cfg):
    time.sleep(3)
    LOG.info("═" * 60)
    LOG.info("  EVOLUTION ENGINE — STARTING")
    LOG.info("═" * 60)

    data = fetch_data(cfg)
    if "gold" not in data: LOG.error("No gold data"); return
    features_full, gold = build_features(data, cfg)
    target = build_target(gold, delay=cfg.execution_delay_days)
    gold_ret = gold.pct_change()

    audit = sign_audit(features_full, gold_ret)
    BUS.update(sign_audit=audit)
    good_features = [f for f, r in audit.items() if r["match"]]
    bad_features = [f for f, r in audit.items() if not r["match"]]
    LOG.info(f"Sign audit: {len(good_features)} match, {len(bad_features)} violations")

    population = []
    for i in range(cfg.population_size):
        c = random_config(cfg)
        r = random.random()
        if r < 0.3: fs = None
        elif r < 0.6: fs = random_feature_subset()
        else: fs = good_features if len(good_features) >= 8 else None
        population.append((c, fs))

    BUS.update(status="evolving")
    experiment_id = 0

    for gen in range(1, cfg.max_generations + 1):
        BUS.current_gen = gen
        LOG.info(f"\n{'━'*60}")
        LOG.info(f"  GENERATION {gen}")
        LOG.info(f"{'━'*60}")
        BUS.update(generation=gen, status="evolving")

        gen_results = []
        for idx, (c, fs) in enumerate(population):
            experiment_id += 1
            label = config_label(c, fs)
            LOG.info(f"\n  ▸ Experiment #{experiment_id} — {label}")
            BUS.update(experiment=experiment_id, config_label=label,
                       config_details=config_details(c, fs),
                       fold_results=[], equity_curve=[], progress=0,
                       current_step=f"gen{gen} exp{idx+1}/{len(population)}")
            try:
                t0 = time.time()
                folds, equity, fi = walk_forward(features_full, target, gold, c,
                                                  feature_subset=fs, emit_folds=True)
                gate = validation_gate(folds, c)
                elapsed = time.time() - t0
                BUS.update(fold_results=folds[-50:], equity_curve=equity[-200:],
                           gate_result=gate, feature_importance=fi, progress=100,
                           current_step=f"gen{gen} exp{idx+1} DONE")
                exp_result = {
                    "id": experiment_id, "generation": gen, "label": label,
                    "config": config_details(c, fs),
                    "score": gate["score"], "raw_sharpe": gate["raw_sharpe"],
                    "deflated_sharpe": gate["deflated_sharpe"],
                    "fold_failure_rate": gate["fold_failure_rate"],
                    "total_return": gate["total_return"],
                    "num_folds": gate["num_folds"], "passed": gate["passed"],
                    "elapsed": round(elapsed, 1),
                    "equity_final": equity[-1] if equity else 1.0,
                }
                gen_results.append((exp_result, c, fs))
                BUS.add_experiment(exp_result)
                status = "✓ PASS" if gate["passed"] else "✗"
                LOG.info(f"    Score={gate['score']:+.4f}  Sharpe={gate['raw_sharpe']:+.4f}  "
                         f"DSR={gate['deflated_sharpe']:.4f}  FailRate={gate['fold_failure_rate']:.2f}  "
                         f"Return={gate['total_return']:+.4f}  {status}  [{elapsed:.1f}s]")
            except Exception as e:
                LOG.error(f"    Experiment failed: {e}")
                gen_results.append(({"id":experiment_id,"score":-999,"label":label,
                                     "generation":gen,"passed":False}, c, fs))

        # Selection + Breeding
        gen_results.sort(key=lambda x: x[0]["score"], reverse=True)
        LOG.info(f"\n  Generation {gen} Rankings:")
        for rank, (res, _, _) in enumerate(gen_results):
            LOG.info(f"    #{rank+1} Score={res['score']:+.4f} — {res['label']}")

        new_pop = []
        elites = gen_results[:cfg.elite_keep]
        for res, c, fs in elites:
            new_pop.append((copy.deepcopy(c), fs[:] if fs else None))
            LOG.info(f"  ★ Elite kept: {res['label']} (score={res['score']:+.4f})")

        while len(new_pop) < cfg.population_size:
            parent_res, parent_c, parent_fs = random.choice(elites)
            child_c = mutate_config(parent_c)
            child_fs = parent_fs
            if random.random() < cfg.mutation_rate:
                r = random.random()
                if r < 0.25: child_fs = None
                elif r < 0.5: child_fs = random_feature_subset()
                elif r < 0.75 and len(good_features) >= 8: child_fs = good_features
                else:
                    if parent_fs:
                        child_fs = parent_fs[:]
                        for _ in range(random.randint(1,3)):
                            if random.random() < 0.5 and len(child_fs) > 8:
                                child_fs.remove(random.choice(child_fs))
                            else:
                                cands = [f for f in ALL_FEATURES if f not in child_fs]
                                if cands: child_fs.append(random.choice(cands))
            new_pop.append((child_c, child_fs))
        population = new_pop
        BUS.signal({"type":"generation_complete","generation":gen,
                     "best_score":gen_results[0][0]["score"],
                     "best_label":gen_results[0][0]["label"]})

    LOG.info("\n  EVOLUTION COMPLETE")
    BUS.update(status="complete")


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def metrics_loop():
    while True:
        try:
            snap = {}
            if HAS_PSUTIL:
                snap["cpu"] = psutil.cpu_percent(interval=None)
                snap["cpus"] = psutil.cpu_percent(interval=None, percpu=True)
                m = psutil.virtual_memory()
                snap["mem_pct"] = m.percent; snap["mem_used"] = round(m.used/1e9,1); snap["mem_total"] = round(m.total/1e9,1)
                snap["disk_pct"] = psutil.disk_usage("/").percent
                l1,l5,l15 = os.getloadavg()
                snap["load"] = [round(l1,2),round(l5,2),round(l15,2)]
                snap["uptime"] = int(time.time()-psutil.boot_time())
                top = []
                for p in sorted(psutil.process_iter(["pid","name","cpu_percent","memory_percent"]),
                                key=lambda x: x.info.get("cpu_percent") or 0, reverse=True)[:8]:
                    top.append({"pid":p.info["pid"],"name":p.info["name"],
                                "cpu":round(p.info.get("cpu_percent") or 0,1),
                                "mem":round(p.info.get("memory_percent") or 0,1)})
                snap["procs"] = top
            BUS.metric(snap)
        except: pass
        time.sleep(3)


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(16).hex()
sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
BUS.bind(sio)

HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phoenix v5 Evolution</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.4/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
:root{
  --b0:#050608;--b1:#0a0d13;--b2:#10141c;--b3:#171c28;--b4:#1e2436;
  --bdr:#252d42;--bdr2:#303a54;
  --t0:#e8ecf8;--t1:#8892b0;--t2:#4a5270;
  --gold:#e8b84b;--gold2:#c49a30;--gold-g:rgba(232,184,75,.12);
  --grn:#34d399;--grn-g:rgba(52,211,153,.1);
  --red:#f43f5e;--red-g:rgba(244,63,94,.08);
  --ylw:#fbbf24;--ylw-g:rgba(251,191,36,.08);
  --cyan:#22d3ee;--cyan-g:rgba(34,211,238,.08);
  --vio:#a78bfa;--vio-g:rgba(167,139,250,.08);
  --mono:'IBM Plex Mono',monospace;--sans:'Outfit',system-ui,sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:13px}
body{background:var(--b0);color:var(--t0);font-family:var(--sans);min-height:100vh}
::selection{background:var(--gold);color:#000}
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-thumb{background:var(--b4);border-radius:4px}
.top{position:sticky;top:0;z-index:100;display:flex;align-items:center;gap:12px;
  padding:8px 16px;background:rgba(5,6,8,.9);backdrop-filter:blur(30px);border-bottom:1px solid var(--bdr)}
.top .brand{display:flex;align-items:center;gap:8px;font-family:var(--mono);font-weight:700;font-size:.95rem;color:var(--gold)}
.top .brand .dot{width:8px;height:8px;border-radius:50%;background:var(--grn);box-shadow:0 0 12px var(--grn)}
.dot.evolving{background:var(--gold);box-shadow:0 0 12px var(--gold);animation:glow 1.5s infinite}
.dot.err{background:var(--red);box-shadow:0 0 12px var(--red)}
@keyframes glow{0%,100%{box-shadow:0 0 8px var(--gold)}50%{box-shadow:0 0 20px var(--gold)}}
.top .sep{color:var(--bdr);font-weight:300}
.top .meta{font-size:.68rem;color:var(--t2);font-family:var(--mono)}
.top .spacer{flex:1}
.pill{font-family:var(--mono);font-size:.62rem;padding:3px 10px;border-radius:10px;display:inline-block}
.pill-g{background:var(--grn-g);color:var(--grn)}.pill-r{background:var(--red-g);color:var(--red)}
.pill-y{background:var(--ylw-g);color:var(--ylw)}.pill-d{background:var(--b3);color:var(--t2)}
.pill-v{background:var(--vio-g);color:var(--vio)}
.main{display:grid;grid-template-columns:1fr 340px;grid-template-rows:auto auto 1fr;min-height:calc(100vh - 41px)}
.strip{grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1px;background:var(--bdr)}
.mc{background:var(--b1);padding:12px 14px;display:flex;flex-direction:column;gap:3px}
.mc .l{font-size:.55rem;text-transform:uppercase;letter-spacing:2px;color:var(--t2);font-weight:700}
.mc .v{font-family:var(--mono);font-size:1.35rem;font-weight:700;line-height:1}
.mc .s{font-size:.62rem;color:var(--t2);font-family:var(--mono)}
.mc .bar{height:3px;background:var(--b4);border-radius:2px;margin-top:3px;overflow:hidden}
.mc .bar div{height:100%;border-radius:2px;transition:width .4s}
.gate{grid-column:1/-1;display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--bdr)}
.gc{background:var(--b1);padding:10px 14px;text-align:center}
.gc .gl{font-size:.52rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--t2);font-weight:700;margin-bottom:3px}
.gc .gv{font-family:var(--mono);font-size:1.1rem;font-weight:700}
.gc .gs{font-size:.58rem;margin-top:2px}
.charts{background:var(--b1);border-right:1px solid var(--bdr);display:flex;flex-direction:column;min-height:0}
.chart-tabs{display:flex;gap:2px;padding:6px 12px 0;border-bottom:1px solid var(--bdr);flex-shrink:0}
.ctab{font-family:var(--mono);font-size:.64rem;padding:5px 12px;background:transparent;color:var(--t2);
  border:none;cursor:pointer;border-radius:5px 5px 0 0;transition:all .15s}
.ctab:hover{color:var(--t1);background:var(--b2)}
.ctab.active{color:var(--gold);background:var(--b2);box-shadow:inset 0 2px 0 var(--gold)}
.chart-body{flex:1;padding:8px 12px;min-height:0;position:relative}
.chart-body canvas{width:100%!important;max-height:280px}
.sidebar{display:flex;flex-direction:column;min-height:0;background:var(--b1)}
.lb{flex-shrink:0;max-height:45vh;overflow-y:auto;border-bottom:1px solid var(--bdr)}
.lb-head{position:sticky;top:0;background:var(--b1);padding:8px 12px;border-bottom:1px solid var(--bdr);
  display:flex;align-items:center;gap:8px}
.lb-head h2{font-size:.68rem;text-transform:uppercase;letter-spacing:1.2px;color:var(--t1);font-weight:600}
.lb-row{display:grid;grid-template-columns:24px 1fr 60px 52px;gap:4px;align-items:center;
  padding:5px 12px;border-bottom:1px solid rgba(37,45,66,.3);font-family:var(--mono);font-size:.66rem;transition:background .1s}
.lb-row:hover{background:var(--b2)}
.lb-rank{color:var(--t2);text-align:center;font-weight:600}
.lb-rank.top{color:var(--gold)}
.lb-label{color:var(--t1);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:.6rem}
.lb-score{text-align:right;font-weight:600}
.lb-gen{color:var(--t2);text-align:center;font-size:.58rem}
.logs{flex:1;display:flex;flex-direction:column;min-height:0}
.logs-head{position:sticky;top:0;padding:6px 12px;border-bottom:1px solid var(--bdr);display:flex;align-items:center;gap:8px;flex-shrink:0;background:var(--b1)}
.logs-head h2{font-size:.68rem;text-transform:uppercase;letter-spacing:1.2px;color:var(--t1);font-weight:600}
.logs-head input{flex:1;background:var(--b2);border:1px solid var(--bdr);border-radius:4px;padding:4px 8px;
  font-family:var(--mono);font-size:.68rem;color:var(--t0);outline:0}
.logs-head input:focus{border-color:var(--gold)}
.log-view{flex:1;overflow-y:auto;font-family:var(--mono);font-size:.66rem;line-height:1.6;min-height:0}
.ll{display:flex;padding:1px 12px;border-left:2px solid transparent;transition:background .1s;animation:fadeIn .12s}
.ll:hover{background:var(--b2)}
.ll.error{border-left-color:var(--red);background:var(--red-g)}
.ll.warn{border-left-color:var(--ylw);background:var(--ylw-g)}
.ll .ts{flex-shrink:0;width:62px;color:var(--t2);font-size:.58rem}
.ll .src{flex-shrink:0;width:52px;color:var(--gold);font-weight:500;font-size:.6rem}
.ll .msg{flex:1;white-space:pre-wrap;word-break:break-all}
.ll.error .msg{color:var(--red)}.ll.warn .msg{color:var(--ylw)}
@keyframes fadeIn{from{opacity:0;transform:translateY(-2px)}to{opacity:1}}
.fi-panel{display:none;flex-direction:column;padding:8px 12px;gap:3px;overflow-y:auto}
.fi-panel.active{display:flex}
.fi-r{display:flex;align-items:center;gap:6px}
.fi-r .fn{font-family:var(--mono);font-size:.6rem;color:var(--t1);width:150px;text-align:right;flex-shrink:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fi-r .fb{flex:1;height:5px;background:var(--b4);border-radius:3px;overflow:hidden}
.fi-r .ff{height:100%;border-radius:3px;transition:width .4s}
.fi-r .fv{font-family:var(--mono);font-size:.56rem;color:var(--t2);width:40px;flex-shrink:0}
.fold-ticker{grid-column:1/-1;overflow:hidden;height:28px;background:var(--b1);border-top:1px solid var(--bdr);
  display:flex;align-items:center;padding:0 12px;gap:16px;font-family:var(--mono);font-size:.6rem}
.fold-ticker .ft-item{display:flex;gap:6px;align-items:center;animation:slideIn .3s;flex-shrink:0}
.fold-ticker .ft-id{color:var(--t2)}
.fold-ticker .ft-sharpe{font-weight:600}
@keyframes slideIn{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:none}}
.fold-table-wrap{display:none;overflow:auto;flex:1}
.fold-table-wrap.active{display:block}
table.ft{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.64rem}
.ft th{text-align:left;padding:6px 8px;font-size:.54rem;text-transform:uppercase;letter-spacing:1px;
  color:var(--t2);font-weight:600;border-bottom:1px solid var(--bdr);position:sticky;top:0;background:var(--b1)}
.ft td{padding:4px 8px;border-bottom:1px solid rgba(37,45,66,.3);color:var(--t1)}
.ft tr:hover td{background:var(--b2)}
.pass{color:var(--grn)}.fail{color:var(--red)}
.cfg-panel{display:none;padding:12px;font-family:var(--mono);font-size:.66rem;color:var(--t1);overflow-y:auto}
.cfg-panel.active{display:block}
.cfg-panel .cfg-row{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(37,45,66,.2)}
.cfg-panel .cfg-k{color:var(--t2)}.cfg-panel .cfg-v{color:var(--t0);font-weight:500}
@media(max-width:900px){.main{grid-template-columns:1fr}.sidebar{max-height:50vh}}
</style>
</head><body>
<header class="top">
  <div class="brand"><span class="dot" id="dot"></span><span>PHOENIX</span></div>
  <span class="sep">|</span><span class="meta">v5 EVOLUTION · GC=F</span>
  <div class="spacer"></div>
  <span class="pill pill-d" id="genPill">GEN 0</span>
  <span class="pill pill-d" id="expPill">EXP 0</span>
  <span class="pill pill-d" id="foldsPill">0 folds</span>
  <span class="pill pill-d" id="bestPill">best —</span>
  <span class="pill pill-d" id="connPill">—</span>
</header>
<div class="fold-ticker" id="ticker"></div>
<div class="main">
  <div class="strip">
    <div class="mc"><span class="l">CPU</span><span class="v" id="mCpu">—</span>
      <div class="bar"><div id="mCpuB" style="width:0%;background:var(--cyan)"></div></div>
      <span class="s" id="mLoad">—</span></div>
    <div class="mc"><span class="l">Memory</span><span class="v" id="mMem">—</span>
      <div class="bar"><div id="mMemB" style="width:0%;background:var(--grn)"></div></div>
      <span class="s" id="mMemS">—</span></div>
    <div class="mc"><span class="l">Generation</span><span class="v" id="mGen">0</span>
      <span class="s" id="mStep">idle</span></div>
    <div class="mc"><span class="l">Experiments</span><span class="v" id="mExp">0</span>
      <span class="s" id="mExpS">—</span></div>
    <div class="mc"><span class="l">Total Folds</span><span class="v" id="mFolds">0</span></div>
    <div class="mc"><span class="l">Best Score</span><span class="v" id="mBest">—</span>
      <span class="s" id="mBestL">—</span></div>
    <div class="mc"><span class="l">Uptime</span><span class="v" id="mUp">—</span></div>
  </div>
  <div class="gate">
    <div class="gc"><div class="gl">Gate</div><div class="gv" id="gAll">—</div><div class="gs pill pill-d" id="gAllS">waiting</div></div>
    <div class="gc"><div class="gl">Raw Sharpe</div><div class="gv" id="gS">—</div><div class="gs pill pill-d" id="gSS">—</div></div>
    <div class="gc"><div class="gl">Deflated SR</div><div class="gv" id="gD">—</div><div class="gs pill pill-d" id="gDS">—</div></div>
    <div class="gc"><div class="gl">Fold Fail</div><div class="gv" id="gF">—</div><div class="gs pill pill-d" id="gFS">—</div></div>
  </div>
  <div class="charts">
    <div class="chart-tabs">
      <button class="ctab active" onclick="showTab('equity')">Equity Curve</button>
      <button class="ctab" onclick="showTab('sharpe')">Sharpe History</button>
      <button class="ctab" onclick="showTab('features')">Features</button>
      <button class="ctab" onclick="showTab('foldtable')">Fold Table</button>
      <button class="ctab" onclick="showTab('config')">Config</button>
    </div>
    <div class="chart-body" id="chartEquity"><canvas id="eqChart"></canvas></div>
    <div class="chart-body" id="chartSharpe" style="display:none"><canvas id="shChart"></canvas></div>
    <div class="fi-panel" id="chartFeatures"></div>
    <div class="fold-table-wrap" id="chartFoldtable">
      <table class="ft"><thead><tr><th>#</th><th>Period</th><th>Sharpe</th><th>Ret</th><th>MDD</th><th>Acc</th><th>CumRet</th><th>✓</th></tr></thead>
        <tbody id="foldBody"></tbody></table>
    </div>
    <div class="cfg-panel" id="chartConfig"></div>
  </div>
  <div class="sidebar">
    <div class="lb" id="lbWrap">
      <div class="lb-head"><h2>Leaderboard</h2><span class="pill pill-d" id="lbCount">0</span></div>
      <div id="lbBody"></div>
    </div>
    <div class="logs">
      <div class="logs-head"><h2>Logs</h2><input id="logF" placeholder="filter…" oninput="filterLogs()"></div>
      <div class="log-view" id="logView"></div>
    </div>
  </div>
</div>
<script>
const S=io();let logs=[],logFilter='',autoScroll=true,totalFolds=0,eqChart,shChart,eqData=[],shData=[],bestEver=-999;
const C={grn:'#34d399',red:'#f43f5e',gold:'#e8b84b',cyan:'#22d3ee',vio:'#a78bfa'};
S.on('connect',()=>{g('connPill').textContent='LIVE';g('connPill').className='pill pill-g';S.emit('sync')});
S.on('disconnect',()=>{g('connPill').textContent='OFF';g('connPill').className='pill pill-r'});
function initCharts(){
  const b={responsive:true,maintainAspectRatio:false,animation:{duration:0},
    plugins:{legend:{display:false}},scales:{x:{display:true,grid:{color:'#1e243622'},ticks:{color:'#4a5270',font:{size:9,family:'IBM Plex Mono'}}},
    y:{grid:{color:'#1e243644'},ticks:{color:'#4a5270',font:{size:9,family:'IBM Plex Mono'}}}}};
  eqChart=new Chart(g('eqChart'),{type:'line',data:{labels:[],datasets:[{data:[],borderColor:C.gold,borderWidth:1.5,pointRadius:0,fill:{target:'origin',above:'rgba(232,184,75,.06)'},tension:.3}]},
    options:{...b,scales:{...b.scales,y:{...b.scales.y,title:{display:true,text:'Equity',color:'#4a5270',font:{size:9}}}}}});
  shChart=new Chart(g('shChart'),{type:'bar',data:{labels:[],datasets:[{data:[],backgroundColor:[],borderRadius:2,barPercentage:.7}]},
    options:{...b,scales:{...b.scales,y:{...b.scales.y,title:{display:true,text:'Sharpe',color:'#4a5270',font:{size:9}}}}}});
}
function showTab(t){
  document.querySelectorAll('.ctab').forEach(b=>b.classList.remove('active'));
  event.target.classList.add('active');
  ['Equity','Sharpe','Features','Foldtable','Config'].forEach(n=>{const e=g('chart'+n);if(e){e.style.display='none';e.classList.remove('active')}});
  const p=g('chart'+t.charAt(0).toUpperCase()+t.slice(1));if(p){p.style.display='';p.classList.add('active')}
}
S.on('fold_tick',(d)=>{
  totalFolds=d.total_folds||totalFolds+1;g('mFolds').textContent=totalFolds;g('foldsPill').textContent=totalFolds+' folds';
  const tk=g('ticker'),item=document.createElement('span');item.className='ft-item';
  item.innerHTML=`<span class="ft-id">#${d.fold_id}</span><span class="ft-sharpe ${d.sharpe>0?'pass':'fail'}">${d.sharpe>0?'+':''}${d.sharpe.toFixed(2)}</span><span style="color:var(--t2)">${(d.return*100).toFixed(1)}%</span>`;
  tk.appendChild(item);if(tk.children.length>30)tk.removeChild(tk.firstChild);tk.scrollLeft=tk.scrollWidth;
  shData.push(d.sharpe);if(shData.length>300)shData=shData.slice(-300);
});
S.on('evo_state',(s)=>{
  g('dot').className='dot '+(s.status==='evolving'?'evolving':'');
  g('mGen').textContent=s.generation||0;g('genPill').textContent='GEN '+(s.generation||0);
  g('mExp').textContent=s.experiment||0;g('expPill').textContent='EXP '+(s.experiment||0);
  g('mStep').textContent=s.current_step||'idle';
  if(s.config_label)g('mExpS').textContent=s.config_label;
  if(s.equity_curve&&s.equity_curve.length){eqData=s.equity_curve;eqChart.data.labels=eqData.map((_,i)=>i);eqChart.data.datasets[0].data=eqData;eqChart.update('none')}
  if(shData.length){shChart.data.labels=shData.map((_,i)=>i);shChart.data.datasets[0].data=shData;shChart.data.datasets[0].backgroundColor=shData.map(v=>v>0?C.grn+'88':C.red+'88');shChart.update('none')}
  if(s.fold_results){g('foldBody').innerHTML=s.fold_results.map(f=>`<tr><td>${f.fold_id}</td><td style="font-size:.58rem">${f.test_start}→${f.test_end}</td><td class="${f.sharpe>0?'pass':'fail'}">${f.sharpe>0?'+':''}${f.sharpe.toFixed(3)}</td><td class="${f.return>0?'pass':'fail'}">${(f.return*100).toFixed(2)}%</td><td>${(f.mdd*100).toFixed(1)}%</td><td>${(f.accuracy*100).toFixed(0)}%</td><td>${f.cumulative_return?f.cumulative_return.toFixed(3):'—'}</td><td class="${f.passed?'pass':'fail'}">${f.passed?'✓':'✗'}</td></tr>`).join('')}
  if(s.gate_result){const r=s.gate_result;g('gAll').textContent=r.passed?'PASS':'FAIL';g('gAllS').textContent=r.passed?'Lane B Unlocked':'Score: '+r.score;g('gAllS').className='gs pill '+(r.passed?'pill-g':'pill-r');setG('gS','gSS',r.raw_sharpe,r.check_sharpe);setG('gD','gDS',r.deflated_sharpe,r.check_dsr);setG('gF','gFS',r.fold_failure_rate,r.check_folds)}
  if(s.feature_importance&&Object.keys(s.feature_importance).length){const c=g('chartFeatures'),e=Object.entries(s.feature_importance),mx=e[0]?e[0][1]:1;c.innerHTML=e.slice(0,20).map(([n,v])=>{const p=mx>0?(v/mx*100):0;const cl=p>70?C.gold:p>40?C.cyan:C.vio;return`<div class="fi-r"><span class="fn">${n}</span><div class="fb"><div class="ff" style="width:${p}%;background:${cl}"></div></div><span class="fv">${v.toFixed(4)}</span></div>`}).join('')}
  if(s.config_details&&Object.keys(s.config_details).length){const c=g('chartConfig');c.innerHTML=Object.entries(s.config_details).filter(([k])=>k!=='features').map(([k,v])=>`<div class="cfg-row"><span class="cfg-k">${k}</span><span class="cfg-v">${v}</span></div>`).join('');if(s.config_details.features)c.innerHTML+=`<div class="cfg-row"><span class="cfg-k">features (${s.config_details.features.length})</span><span class="cfg-v" style="font-size:.58rem;word-break:break-all">${s.config_details.features.join(', ')}</span></div>`}
});
function setG(vi,si,val,pass){g(vi).textContent=val!=null?val.toFixed(4):'—';g(si).textContent=pass?'PASS':'FAIL';g(si).className='gs pill '+(pass?'pill-g':'pill-r')}
S.on('leaderboard',(lb)=>{
  g('lbCount').textContent=lb.length;
  g('lbBody').innerHTML=lb.map((e,i)=>`<div class="lb-row"><span class="lb-rank ${i<3?'top':''}">${i+1}</span><span class="lb-label" title="${e.label}">${e.label}</span><span class="lb-score ${e.score>0?'pass':'fail'}">${e.score>0?'+':''}${e.score.toFixed(3)}</span><span class="lb-gen">G${e.generation}</span></div>`).join('');
  if(lb[0]){bestEver=Math.max(bestEver,lb[0].score);g('mBest').textContent=bestEver>-100?bestEver.toFixed(4):'—';g('mBestL').textContent=lb[0].label;g('bestPill').textContent='best '+bestEver.toFixed(3);g('bestPill').className='pill '+(bestEver>0?'pill-g':'pill-r')}
});
S.on('experiment',(e)=>{g('mExp').textContent=e.id});
S.on('log',(d)=>{logs.push(d);if(logs.length>2000)logs=logs.slice(-1500);if(matchF(d))appendLog(d)});
function appendLog(d){const v=g('logView'),div=document.createElement('div');div.className='ll '+(d.level||'');div.innerHTML=`<span class="ts">${d.ts?new Date(d.ts).toLocaleTimeString('en',{hour12:false}):''}</span><span class="src">${d.source||''}</span><span class="msg">${esc(d.msg||'')}</span>`;v.appendChild(div);if(v.children.length>1200)for(let i=0;i<200;i++)v.removeChild(v.firstChild);if(autoScroll)v.scrollTop=v.scrollHeight}
function matchF(d){if(!logFilter)return true;const t=logFilter.toLowerCase();return(d.msg||'').toLowerCase().includes(t)||(d.source||'').toLowerCase().includes(t)}
function filterLogs(){logFilter=g('logF').value;g('logView').innerHTML='';logs.filter(matchF).forEach(appendLog)}
g('logView').addEventListener('scroll',function(){autoScroll=this.scrollTop+this.clientHeight>=this.scrollHeight-30});
S.on('sys_metrics',(d)=>{
  if(d.cpu!=null){g('mCpu').textContent=d.cpu+'%';g('mCpuB').style.width=d.cpu+'%';g('mCpuB').style.background=d.cpu>80?C.red:d.cpu>50?C.gold:C.cyan}
  if(d.mem_pct!=null){g('mMem').textContent=d.mem_pct+'%';g('mMemB').style.width=d.mem_pct+'%';g('mMemS').textContent=(d.mem_used||0)+'/'+(d.mem_total||0)+' GB'}
  if(d.load)g('mLoad').textContent='load '+d.load.join(' · ');
  if(d.uptime!=null){const h=Math.floor(d.uptime/3600),m=Math.floor(d.uptime%3600/60);g('mUp').textContent=h+'h '+m+'m'}
});
S.on('full_sync',(d)=>{
  if(d.logs)d.logs.forEach(l=>{logs.push(l);appendLog(l)});
  if(d.evo_state){S.listeners('evo_state').forEach(fn=>fn(d.evo_state))}
  if(d.leaderboard&&d.leaderboard.length){S.listeners('leaderboard').forEach(fn=>fn(d.leaderboard))}
  totalFolds=d.total_folds||0;g('mFolds').textContent=totalFolds;g('foldsPill').textContent=totalFolds+' folds';
});
function g(id){return document.getElementById(id)}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
window.addEventListener('load',initCharts);
</script></body></html>"""


@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/api/logs")
def api_logs(): return jsonify(list(BUS.logs)[-200:])

@app.route("/api/leaderboard")
def api_lb(): return jsonify(BUS.leaderboard)

@app.route("/api/experiments")
def api_exp(): return jsonify(BUS.experiments[-50:])

@app.route("/api/state")
def api_state(): return jsonify(BUS.evolution_state)

@sio.on("connect")
def on_connect(): emit("connected",{"ok":True})

@sio.on("sync")
def on_sync():
    emit("full_sync",{
        "logs": list(BUS.logs)[-200:],
        "evo_state": BUS.evolution_state,
        "leaderboard": BUS.leaderboard,
        "total_folds": BUS.total_folds_run,
    })
    emit("evo_state", BUS.evolution_state)
    if BUS.leaderboard: emit("leaderboard", BUS.leaderboard)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║   ██████╗ ██╗  ██╗ ██████╗ ███████╗███╗   ██╗██╗██╗  ██╗       ║
    ║   ██╔══██╗██║  ██║██╔═══██╗██╔════╝████╗  ██║██║╚██╗██╔╝       ║
    ║   ██████╔╝███████║██║   ██║█████╗  ██╔██╗ ██║██║ ╚███╔╝        ║
    ║   ██╔═══╝ ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║██║ ██╔██╗        ║
    ║   ██║     ██║  ██║╚██████╔╝███████╗██║ ╚████║██║██╔╝ ██╗       ║
    ║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝       ║
    ║                                                                  ║
    ║   v5 EVOLUTION — Self-Improving Gold Futures Engine              ║
    ║   Continuously mutates · breeds · selects · improves             ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    threading.Thread(target=metrics_loop, daemon=True).start()
    LOG.info("System metrics started")
    threading.Thread(target=evolution_loop, args=(CFG,), daemon=True).start()
    LOG.info("Evolution engine started")
    LOG.info(f"Dashboard → http://0.0.0.0:{CFG.flask_port}")
    sio.run(app, host=CFG.flask_host, port=CFG.flask_port, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
