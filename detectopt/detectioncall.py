import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm


class Struct:
    """Lightweight class for dot-notation, matching the rest of the framework."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def merge_params(s: Any, snew: Any) -> Any:
    """Merge fields from snew into s, matching MATLAB mergeParams behavior."""
    if s is None:
        s = Struct()
    if snew is None:
        return s
    for k, v in vars(snew).items():
        setattr(s, k, v)
    return s


def _as_array(x: Any) -> np.ndarray:
    return np.asarray(x)


def _as_flat_bool(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=bool).reshape(-1)


def _as_flat_float(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _first_true_idx(mask: np.ndarray) -> Optional[int]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def _matrix_power(a: np.ndarray, k: int) -> np.ndarray:
    if a.ndim == 0:
        return np.asarray(float(a) ** k)
    if a.shape == (1, 1):
        return np.asarray([[float(a[0, 0]) ** k]])
    return np.linalg.matrix_power(a, k)


def _to_float(x: Any) -> float:
    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def _select_constraint_index(fp: np.ndarray, pmd: np.ndarray, params: Any) -> int:
    if params.consttype == 1:
        return int(np.argmin(np.abs(fp - params.maxfprate)))
    if params.consttype == 2:
        return int(np.argmin(np.abs(pmd - params.maxpmd)))
    return int(np.argmin(np.abs(pmd - fp)))


def _predictive_alarm_series(obs_segment: Any, lds: Any, dstep: int, laopt: float) -> Tuple[np.ndarray, np.ndarray]:
    data = _as_flat_float(obs_segment.data)
    adl = _as_array(lds.adl)
    cdl = _as_array(lds.cdl)
    kfgain = _as_array(lds.kfgain)
    adl_d = _matrix_power(adl, int(dstep))

    xhatpost = _as_array(lds.initx0)
    predict = np.zeros(data.size, dtype=float)
    alarms = np.zeros(data.size, dtype=bool)

    for k in range(data.size):
        xhat = adl @ xhatpost
        res_k = _to_float(data[k] - (cdl @ xhat))
        xhatpost = xhat + kfgain * res_k
        pred_k = _to_float(cdl @ adl_d @ xhatpost)
        predict[k] = pred_k
        alarms[k] = abs(pred_k) > laopt

    return predict, alarms


def detectioncall(
    obsval,
    obstest,
    params,
    detect_idx: int,
    levelparams=None,
    lds_params=None,
):
    """
    Serial translation of ACCEPT/detectopt/detectioncall.m.

    Distributed/PCT/MCR branches are intentionally omitted.
    """
    et = time.process_time()
    stats = Struct()
    alarms: List[np.ndarray] = []

    if detect_idx == 1:
        print("Now optimizing redline alarm system and generating results using training data")
        stats.redtrain = Struct(x=levelparams.L)

        # TODO: implement detectopt.predopt.redlineopt
        from detectopt.predopt import redlineopt

        auctemp, fptemp, tptemp, patemp, pcatemp, lavaltemp = [], [], [], [], [], []
        for lds in lds_params:
            auc_i, fp_i, tp_i, pa_i, pca_i, laval_i = redlineopt(params, lds, levelparams.dstep, stats.redtrain.x)
            auctemp.append(auc_i)
            fptemp.append(_as_flat_float(fp_i))
            tptemp.append(_as_flat_float(tp_i))
            patemp.append(pa_i)
            pcatemp.append(pca_i)
            lavaltemp.append(_as_flat_float(laval_i))

        modord_idx = int(np.argmax(np.asarray(auctemp, dtype=float)))
        stats.redtrain.auctemp = np.asarray(auctemp, dtype=float)
        stats.redtrain.fptemp = fptemp
        stats.redtrain.tptemp = tptemp
        stats.redtrain.patemp = patemp
        stats.redtrain.pcatemp = pcatemp
        stats.redtrain.Lavaltemp = lavaltemp

        stats.redtrain.auc = stats.redtrain.auctemp[modord_idx]
        stats.redtrain.fp = stats.redtrain.fptemp[modord_idx]
        stats.redtrain.tp = stats.redtrain.tptemp[modord_idx]
        stats.redtrain.pa = stats.redtrain.patemp[modord_idx]
        stats.redtrain.pca = stats.redtrain.pcatemp[modord_idx]
        stats.redtrain.Laval = stats.redtrain.Lavaltemp[modord_idx]

        if params.consttype == 1:
            la_idx = int(np.argmin(np.abs(stats.redtrain.fp - params.maxfprate)))
        elif params.consttype == 2:
            la_idx = int(np.argmin(np.abs(1.0 - stats.redtrain.tp - params.maxpmd)))
        else:
            la_idx = int(np.argmin(np.abs(1.0 - stats.redtrain.tp - stats.redtrain.fp)))
        stats.redtrain.Laopt = float(stats.redtrain.Laval[la_idx])

        for obs in obstest:
            alarms.append(np.abs(_as_flat_float(obs.data)) > stats.redtrain.Laopt)

    elif detect_idx == 2:
        print("Now optimizing redline alarm system and generating results using validation data")
        stats.redval = Struct(x=levelparams.L)

        # TODO: implement detectopt.predopt.exceedvalidation
        from detectopt.predopt import exceedvalidation

        stats.redval.auc, stats.redval.rocdata = exceedvalidation(stats.redval.x, levelparams.dstep, obsval)

        avg_stats = stats.redval.rocdata.avg_stats
        laval_idx = _select_constraint_index(
            _as_flat_float(avg_stats.fprate),
            _as_flat_float(avg_stats.pmd),
            params,
        )
        stats.redval.Laopt = float(_as_flat_float(avg_stats.thresh)[laval_idx])

        for obs in obstest:
            alarms.append(np.abs(_as_flat_float(obs.data)) > stats.redval.Laopt)

    elif detect_idx == 3:
        print("Now optimizing predictive alarm system and generating results using training data")
        stats.predtrain = Struct(x=levelparams.L, predict=[])

        # TODO: implement detectopt.predopt.predtrainopt
        from detectopt.predopt import predtrainopt

        auctemp, fptemp, tptemp, patemp, pcatemp, lavaltemp = [], [], [], [], [], []
        for lds in lds_params:
            auc_i, fp_i, tp_i, pa_i, pca_i, laval_i = predtrainopt(params, lds, levelparams.dstep, stats.predtrain.x)
            auctemp.append(auc_i)
            fptemp.append(_as_flat_float(fp_i))
            tptemp.append(_as_flat_float(tp_i))
            patemp.append(pa_i)
            pcatemp.append(pca_i)
            lavaltemp.append(_as_flat_float(laval_i))

        modord_idx = int(np.argmax(np.asarray(auctemp, dtype=float)))
        stats.predtrain.auctemp = np.asarray(auctemp, dtype=float)
        stats.predtrain.fptemp = fptemp
        stats.predtrain.tptemp = tptemp
        stats.predtrain.patemp = patemp
        stats.predtrain.pcatemp = pcatemp
        stats.predtrain.Lavaltemp = lavaltemp

        stats.predtrain.auc = stats.predtrain.auctemp[modord_idx]
        stats.predtrain.fp = stats.predtrain.fptemp[modord_idx]
        stats.predtrain.tp = stats.predtrain.tptemp[modord_idx]
        stats.predtrain.pa = stats.predtrain.patemp[modord_idx]
        stats.predtrain.pca = stats.predtrain.pcatemp[modord_idx]
        stats.predtrain.Laval = stats.predtrain.Lavaltemp[modord_idx]

        if params.consttype == 1:
            la_idx = int(np.argmin(np.abs(stats.predtrain.fp - params.maxfprate)))
        elif params.consttype == 2:
            la_idx = int(np.argmin(np.abs(1.0 - stats.predtrain.tp - params.maxpmd)))
        else:
            la_idx = int(np.argmin(np.abs(1.0 - stats.predtrain.tp - stats.predtrain.fp)))
        stats.predtrain.Laopt = float(stats.predtrain.Laval[la_idx])

        chosen_lds = lds_params[modord_idx]
        for obs in obstest:
            pred, alarm = _predictive_alarm_series(obs, chosen_lds, int(levelparams.dstep), stats.predtrain.Laopt)
            stats.predtrain.predict.append(pred)
            alarms.append(alarm)

    elif detect_idx == 4:
        print("Now optimizing predictive alarm system and generating results using validation data")
        stats.predval = Struct(x=levelparams.L, predict=[])

        # TODO: implement detectopt.predopt.predlineopt
        from detectopt.predopt import predlineopt

        auctemp, rocdatatemp = [], []
        for lds in lds_params:
            auc_i, roc_i = predlineopt(stats.predval.x, lds, levelparams.dstep, obsval)
            auctemp.append(auc_i)
            rocdatatemp.append(roc_i)

        modord_idx = int(np.argmax(np.asarray(auctemp, dtype=float)))
        stats.predval.auctemp = np.asarray(auctemp, dtype=float)
        stats.predval.rocdatatemp = rocdatatemp
        stats.predval.auc = stats.predval.auctemp[modord_idx]
        stats.predval.rocdata = stats.predval.rocdatatemp[modord_idx]

        avg_stats = stats.predval.rocdata.avg_stats
        laval_idx = _select_constraint_index(
            _as_flat_float(avg_stats.fprate),
            _as_flat_float(avg_stats.pmd),
            params,
        )
        stats.predval.Laopt = float(_as_flat_float(avg_stats.thresh)[laval_idx])

        chosen_lds = lds_params[modord_idx]
        for obs in obstest:
            pred, alarm = _predictive_alarm_series(obs, chosen_lds, int(levelparams.dstep), stats.predval.Laopt)
            stats.predval.predict.append(pred)
            alarms.append(alarm)

    elif detect_idx == 5:
        print("Now optimizing optimal alarm system parameters and generating results using training data")
        stats.opttrain = Struct(x=levelparams.L, predict=[], condeventprob=[], alarms=[])

        # TODO: implement detectopt.opt.detectopt
        from detectopt.opt.detectopt import detectopt as opt_detectopt
        # TODO: implement detectopt.opt.createVcondeventmatrix
        from detectopt.opt.createVcondeventmatrix import createVcondeventmatrix
        # TODO: implement detectopt.opt.condeventprob
        from detectopt.opt.condeventprob import condeventprob

        auctemp, fptemp, tptemp, patemp, pcatemp, pbvaltemp = [], [], [], [], [], []
        for lds in lds_params:
            auc_i, fp_i, tp_i, pa_i, pca_i, pb_i = opt_detectopt(params, lds, levelparams.dstep, stats.opttrain.x)
            auctemp.append(auc_i)
            fptemp.append(_as_flat_float(fp_i))
            tptemp.append(_as_flat_float(tp_i))
            patemp.append(pa_i)
            pcatemp.append(pca_i)
            pbvaltemp.append(_as_flat_float(pb_i))

        modord_idx = int(np.argmax(np.asarray(auctemp, dtype=float)))
        stats.opttrain.auctemp = np.asarray(auctemp, dtype=float)
        stats.opttrain.fptemp = fptemp
        stats.opttrain.tptemp = tptemp
        stats.opttrain.patemp = patemp
        stats.opttrain.pcatemp = pcatemp
        stats.opttrain.pbvaltemp = pbvaltemp

        stats.opttrain.auc = stats.opttrain.auctemp[modord_idx]
        stats.opttrain.fp = stats.opttrain.fptemp[modord_idx]
        stats.opttrain.tp = stats.opttrain.tptemp[modord_idx]
        stats.opttrain.pa = stats.opttrain.patemp[modord_idx]
        stats.opttrain.pca = stats.opttrain.pcatemp[modord_idx]
        stats.opttrain.pbval = stats.opttrain.pbvaltemp[modord_idx]

        chosen_lds = lds_params[modord_idx]
        chosen_lds.fixed = stats.opttrain.x
        chosen_lds.dstep = levelparams.dstep

        if getattr(params, "flag", 0) == 1:
            chosen_lds.Vcondevent = createVcondeventmatrix(chosen_lds)
        else:
            chosen_lds.pccond = condeventprob(
                chosen_lds,
                stats.opttrain.x,
                levelparams.dstep,
                np.zeros((int(levelparams.dstep),), dtype=float),
            )

        if params.consttype == 1:
            pb_idx = int(np.argmin(np.abs(stats.opttrain.fp - params.maxfprate)))
        elif params.consttype == 2:
            pb_idx = int(np.argmin(np.abs(1.0 - stats.opttrain.tp - params.maxpmd)))
        else:
            pb_idx = int(np.argmin(np.abs(1.0 - stats.opttrain.tp - stats.opttrain.fp)))
        stats.opttrain.pbopt = float(stats.opttrain.pbval[pb_idx])

        # Keep parity with MATLAB behavior for optional feasibility adjustment hooks.
        stats.opttrain.chgflag = False

        adl = _as_array(chosen_lds.adl)
        cdl = _as_array(chosen_lds.cdl)
        kfgain = _as_array(chosen_lds.kfgain)

        for obs in obstest:
            data = _as_flat_float(obs.data)
            xhatpost = _as_array(chosen_lds.initx0)
            pred_mat = np.zeros((int(levelparams.dstep), data.size), dtype=float)
            la_mat = np.zeros((int(levelparams.dstep), data.size), dtype=float)
            cond_prob = np.zeros(data.size, dtype=float)
            alarm = np.zeros(data.size, dtype=bool)

            for k in range(data.size):
                xhat = adl @ xhatpost
                res_k = _to_float(data[k] - (cdl @ xhat))
                xhatpost = xhat + kfgain * res_k

                predict = np.zeros(int(levelparams.dstep), dtype=float)
                laopt = np.zeros(int(levelparams.dstep), dtype=float)
                for j in range(1, int(levelparams.dstep) + 1):
                    predict[j - 1] = _to_float(cdl @ _matrix_power(adl, j) @ xhatpost)
                    vjj = _to_float(chosen_lds.Vcondevent[j - 1, j - 1])
                    laopt[j - 1] = float(chosen_lds.fixed + norm.ppf(stats.opttrain.pbopt) * np.sqrt(vjj))

                pred_mat[:, k] = predict
                la_mat[:, k] = laopt
                cond_prob[k] = _to_float(
                    condeventprob(chosen_lds, stats.opttrain.x, levelparams.dstep, predict)
                )
                alarm[k] = bool(np.any(np.abs(predict) > laopt))

            stats.opttrain.predict.append(pred_mat)
            stats.opttrain.condeventprob.append(cond_prob)
            stats.opttrain.alarms.append(alarm)
            alarms.append(alarm)

    elif detect_idx == 6:
        print("Now optimizing optimal alarm system parameters and generating results using validation data")
        stats.optval = Struct(x=levelparams.L, condeventprob=[])

        # TODO: implement detectopt.opt.optvalidation
        from detectopt.opt.optvalidation import optvalidation
        # TODO: implement detectopt.opt.createVcondeventmatrix
        from detectopt.opt.createVcondeventmatrix import createVcondeventmatrix
        # TODO: implement detectopt.opt.condeventprob
        from detectopt.opt.condeventprob import condeventprob

        auctemp, rocdatatemp = [], []
        for lds in lds_params:
            auc_i, roc_i = optvalidation(stats.optval.x, lds, levelparams.dstep, obsval)
            auctemp.append(auc_i)
            rocdatatemp.append(roc_i)

        modord_idx = int(np.argmax(np.asarray(auctemp, dtype=float)))
        stats.optval.auctemp = np.asarray(auctemp, dtype=float)
        stats.optval.rocdatatemp = rocdatatemp
        stats.optval.auc = stats.optval.auctemp[modord_idx]
        stats.optval.rocdata = stats.optval.rocdatatemp[modord_idx]

        avg_stats = stats.optval.rocdata.avg_stats
        pb_idx = _select_constraint_index(
            _as_flat_float(avg_stats.fprate),
            _as_flat_float(avg_stats.pmd),
            params,
        )
        stats.optval.pbopt = float(_as_flat_float(avg_stats.thresh)[pb_idx])

        chosen_lds = lds_params[modord_idx]
        chosen_lds.fixed = stats.optval.x
        chosen_lds.dstep = levelparams.dstep
        chosen_lds.Vcondevent = createVcondeventmatrix(chosen_lds)

        stats.optval.auc, stats.optval.rocdata = optvalidation(
            stats.optval.x,
            chosen_lds,
            levelparams.dstep,
            obsval,
            stats.optval.pbopt,
        )

        adl = _as_array(chosen_lds.adl)
        cdl = _as_array(chosen_lds.cdl)
        kfgain = _as_array(chosen_lds.kfgain)

        for obs in obstest:
            data = _as_flat_float(obs.data)
            xhatpost = _as_array(chosen_lds.initx0)
            cond_prob = np.zeros(data.size, dtype=float)
            alarm = np.zeros(data.size, dtype=bool)

            for k in range(data.size):
                xhat = adl @ xhatpost
                res_k = _to_float(data[k] - (cdl @ xhat))
                xhatpost = xhat + kfgain * res_k

                predict = np.zeros(int(levelparams.dstep), dtype=float)
                for j in range(1, int(levelparams.dstep) + 1):
                    predict[j - 1] = _to_float(cdl @ _matrix_power(adl, j) @ xhatpost)

                cond_prob[k] = _to_float(
                    condeventprob(chosen_lds, stats.optval.x, levelparams.dstep, predict)
                )
                alarm[k] = cond_prob[k] > stats.optval.pbopt

            stats.optval.condeventprob.append(cond_prob)
            alarms.append(alarm)

        stats.opttest = Struct()
        stats.opttest.auc, stats.opttest.rocdata = optvalidation(
            stats.optval.x,
            chosen_lds,
            levelparams.dstep,
            obstest,
            stats.optval.pbopt,
        )

    elif detect_idx == 7:
        print(
            "Now optimizing SPRT parameters via global optimization toolbox and generating results "
            f"using {len(obsval)}-fold cross validation and test data"
        )
        stats.sprt = Struct(
            Mpos=[], Mneg=[], Vnom=[], Vinv=[], fval=[],
            localhistory=[], globalhistory=[],
            err=None, val=None,
        )

        # TODO: implement detectopt.sprt.sprtsearch
        from detectopt.sprt.sprtsearch import sprtsearch
        # TODO: implement detectopt.sprt.sprtvalidation
        from detectopt.sprt.sprtvalidation import sprtvalidation

        for lds in lds_params:
            vbase = _to_float(lds.cdl @ lds.dare @ np.transpose(lds.cdl) + lds.rvdl)
            params.sprt[0] = params.sprt[2] * np.sqrt(vbase)
            params.sprt[1] = params.sprt[3] * np.sqrt(vbase)

            mpos, mneg, vnom, vinv, fval, localhist, globalhist = sprtsearch(params.sprt, params, lds, obsval)
            stats.sprt.Mpos.append(_to_float(mpos))
            stats.sprt.Mneg.append(_to_float(mneg))
            stats.sprt.Vnom.append(_to_float(vnom))
            stats.sprt.Vinv.append(_to_float(vinv))
            stats.sprt.fval.append(_to_float(fval))
            stats.sprt.localhistory.append(localhist)
            stats.sprt.globalhistory.append(globalhist)

        fvals = np.asarray(stats.sprt.fval, dtype=float)
        modord_idx = int(np.argmin(fvals))
        stats.sprt.modordIdx = modord_idx

        mpos = stats.sprt.Mpos[modord_idx]
        mneg = stats.sprt.Mneg[modord_idx]
        vnom = stats.sprt.Vnom[modord_idx]
        vinv = stats.sprt.Vinv[modord_idx]
        vec = np.array([mpos, mneg, vnom, vinv], dtype=float)

        chosen_lds = lds_params[modord_idx]
        stats.sprt.err, stats.sprt.val = sprtvalidation(vec, chosen_lds, obsval)

        log_thresh = np.log((1.0 - params.maxpmd) / params.maxfprate)
        eps = 1e-300

        stats.res = []
        stats.sprtpos = []
        stats.sprtneg = []
        stats.sprtnom = []
        stats.sprtinv = []
        stats.sprtposalarm = []
        stats.sprtnegalarm = []
        stats.sprtnomalarm = []
        stats.sprtinvalarm = []

        adl = _as_array(chosen_lds.adl)
        cdl = _as_array(chosen_lds.cdl)
        kfgain = _as_array(chosen_lds.kfgain)
        vbase = _to_float(cdl @ chosen_lds.dare @ np.transpose(cdl) + chosen_lds.rvdl)

        for obs in obstest:
            data = _as_flat_float(obs.data)
            xhatpost = _as_array(chosen_lds.initx0)

            res = np.zeros(data.size, dtype=float)
            sprtpos = np.zeros(data.size, dtype=float)
            sprtneg = np.zeros(data.size, dtype=float)
            sprtnom = np.zeros(data.size, dtype=float)
            sprtinv = np.zeros(data.size, dtype=float)

            alarm = np.zeros(data.size, dtype=bool)
            alarm_pos_hist = np.zeros(data.size, dtype=bool)
            alarm_neg_hist = np.zeros(data.size, dtype=bool)
            alarm_nom_hist = np.zeros(data.size, dtype=bool)
            alarm_inv_hist = np.zeros(data.size, dtype=bool)

            for k in range(data.size):
                xhat = adl @ xhatpost
                res[k] = _to_float(data[k] - (cdl @ xhat))
                xhatpost = xhat + kfgain * res[k]

                den = max(norm.pdf(res[k], loc=0.0, scale=np.sqrt(vbase)), eps)
                sprtpos[k] = np.log(max(norm.pdf(res[k], loc=mpos, scale=np.sqrt(vbase)), eps) / den)
                sprtneg[k] = np.log(max(norm.pdf(res[k], loc=-mneg, scale=np.sqrt(vbase)), eps) / den)
                sprtnom[k] = np.log(max(norm.pdf(res[k], loc=0.0, scale=np.sqrt(vnom * vbase)), eps) / den)
                sprtinv[k] = np.log(max(norm.pdf(res[k], loc=0.0, scale=np.sqrt(vbase / vinv)), eps) / den)

                alarm_pos = np.cumsum(sprtpos[: k + 1]) > log_thresh
                alarm_neg = np.cumsum(sprtneg[: k + 1]) > log_thresh
                alarm_nom = np.cumsum(sprtnom[: k + 1]) > log_thresh
                alarm_inv = np.cumsum(sprtinv[: k + 1]) > log_thresh

                alarm_pos_hist[k] = bool(alarm_pos[-1])
                alarm_neg_hist[k] = bool(alarm_neg[-1])
                alarm_nom_hist[k] = bool(alarm_nom[-1])
                alarm_inv_hist[k] = bool(alarm_inv[-1])
                alarm[k] = alarm_pos_hist[k] or alarm_neg_hist[k] or alarm_nom_hist[k] or alarm_inv_hist[k]

            stats.res.append(res)
            stats.sprtpos.append(sprtpos)
            stats.sprtneg.append(sprtneg)
            stats.sprtnom.append(sprtnom)
            stats.sprtinv.append(sprtinv)
            stats.sprtposalarm.append(alarm_pos_hist)
            stats.sprtnegalarm.append(alarm_neg_hist)
            stats.sprtnomalarm.append(alarm_nom_hist)
            stats.sprtinvalarm.append(alarm_inv_hist)
            alarms.append(alarm)

    else:
        raise ValueError(f"Unsupported detect_idx={detect_idx}; expected integer in [1, 7]")

    # Final sample-level metrics (translated from the MATLAB post-switch loop).
    ptest = np.zeros(len(obstest), dtype=float)
    ntest = np.zeros(len(obstest), dtype=float)
    fptest = np.zeros(len(obstest), dtype=float)
    tptest = np.zeros(len(obstest), dtype=float)
    tdtest = np.full(len(obstest), np.nan, dtype=float)

    for l, obs in enumerate(obstest):
        event = _as_flat_bool(obs.event)
        alarm = _as_flat_bool(alarms[l]) if l < len(alarms) else np.zeros_like(event)

        n = min(event.size, alarm.size)
        event = event[:n]
        alarm = alarm[:n]

        ptest[l] = float(np.sum(event))
        ntest[l] = float(n - ptest[l])

        false_alarms = alarm & (~event)
        true_alarms = alarm & event
        fptest[l] = float(np.sum(false_alarms))
        tptest[l] = float(np.sum(true_alarms))

        if ptest[l] > 0:
            te = _first_true_idx(event)
            ta = _first_true_idx(alarm)
            if te is not None and ta is not None and ta < te:
                tdtest[l] = float(te - ta - 1)

    s = Struct()
    if np.sum(ptest) > 0:
        s.recallsamp = float(np.sum(tptest) / np.sum(ptest))
        s.pmdsamp = float(1.0 - s.recallsamp)
    if np.sum(ntest) > 0:
        s.fpratesamp = float(np.sum(fptest) / np.sum(ntest))

    valid_td = tdtest[~np.isnan(tdtest)]
    s.tdsamp = float(np.mean(valid_td)) if valid_td.size > 0 else 0.0

    et = time.process_time() - et

    if detect_idx == 1:
        stats.redtrain = merge_params(stats.redtrain, s)
        print(f"Finished optimizing redline alarm system (by training data) and generating results in {et:.4f} sec")
    elif detect_idx == 2:
        stats.redval = merge_params(stats.redval, s)
        print(f"Finished optimizing redline alarm system (by validation data) and generating results in {et:.4f} sec")
    elif detect_idx == 3:
        stats.predtrain = merge_params(stats.predtrain, s)
        print(f"Finished optimizing predictive alarm system (by training data) and generating results in {et:.4f} sec")
    elif detect_idx == 4:
        stats.predval = merge_params(stats.predval, s)
        print(f"Finished optimizing predictive alarm system (by validation data) and generating results in {et:.4f} sec")
    elif detect_idx == 5:
        stats.opttrain = merge_params(stats.opttrain, s)
        print(f"Finished optimizing optimal alarm system parameters (by training data) and generating results in {et:.4f} sec")
    elif detect_idx == 6:
        stats.optval = merge_params(stats.optval, s)
        print(f"Finished optimizing optimal alarm system parameters (by validation data) and generating results in {et:.4f} sec")
    elif detect_idx == 7:
        stats.sprt = merge_params(stats.sprt, s)
        print(f"Finished optimizing SPRT parameters and generating results in {et:.4f} sec")

    return stats, params
