SELECT
--------------------该块
	SUB.STRIP_NO,
	SUB.STAND_NO,
	--轧制力
	SUB.FORCE + SUB.FORCE FORCE_ACT,
	CF.CORR_FORCE_STAND,
	CF.CORR_FORCE_STAND -PF.CORR_FORCE_STAND CORR_FORCE_DELTA, --
	(CF.ROLL_FORCE - (SUB.FORCE + SUB.FORCE)) / (SUB.FORCE + SUB.FORCE) DETAL_FORCE_CAL,
	(PF.ROLL_FORCE - (SUB.FORCE + SUB.FORCE)) / (SUB.FORCE + SUB.FORCE) DETAL_FORCE_POST,
	PF.STRIP_SPEED - CF.STRIP_SPEED DELTA_SPEED, --
	PF.REL_REDU - CF.REL_REDU DELTA_REDU,
	PF.STRIP_WIDTH ,
	PF.ROLL_DIAM,
	PF.KM,
	PF.CHEM_COEFF,
	--温度
	ACT.FM_TEMP ,--温度实际
	ACT.FM_TEMP - CF.EXIT_TEMP TEMP_DELTA, --控制偏差 可能由水带来
	ACT.FM_TEMP - PF.EXIT_TEMP TEMP_CORR, --模型偏差
	RM.RM_EXIT_TEMP - PDI.R2TEMP RM_TEMP_DELTA, --入口温度和目标偏差
	RM.FET_ACT_TEMP,
	ACT.DESC_F1 + ACT.DESC_F2 DESC_SUM,
	PT.WATER_FLOW,
--	ACT.DESC_F1 - PS.DESCF12_ON + (ACT.DESC_F2 - PS.DESCF23_ON) DELTA_DESC,
	PT.WATER_FLOW - CT.WATER_FLOW DELTA_WATER,
	--辊缝
	SUB .SCREW_DOWN - CG.ROLLGAP_SET GAP_DELTA,
	PG.CORR_ZEROPOINT_USE,
	PG.CORR_ZEROPOINT_USE - PG.CORR_ZEROPOINT DELTA_ZEROPOINT,
	PG.MILLSTRETCH_ROLL,
	PG.MILLSTRETCH_ROLL - CG.MILLSTRETCH_ROLL DELTA_MILL,
	CF.ENTRY_THICK,
	PG.ROLLGAP_OILROLL,
	PG.ROLLWEAR,
	PG.MILLSTRETCH_ZERO,
	CF.ENTRY_TENSION,
	SUB.BEND_FORCE,
--------------遗传
	TIMESTAMPDIFF(2, CHAR(TIMESTAMP(ACT.TOM) - TIMESTAMP('2021-06-01 08:00:00'))) AS UNIX_TIME,
	ACT.FM_THICK - PDI.FMOUTTHICK DELTA_THICK
	--       PS.ROLLEDSEQ,
	--       PS.THICK_CLASS
FROM
	AP.AP_STRIP_PDI AS PDI,
	AP.SCC_ACT_FMSEGACT_SUB AS SUB,
	AP.SCC_ACT_FMSEG AS ACT,
	AP.SCC_CALC_ROLLFORCE AS CF,
	AP.SCC_POST_ROLLFORCE AS PF,
	AP.SCC_CALC_ROLLGAP AS CG,
	AP.SCC_POST_ROLLGAP AS PG,
	AP.SCC_FMSTRIP AS SF,
	AP.SCC_CALC_ENTERTABLE AS RM,
	--     AP.SCC_CALC_PASS AS PS,
	AP.SCC_CALC_TEMP CT,
	AP.SCC_POST_TEMP PT
WHERE
	SUB.SEG_NO = 1
	AND SUB.STRIP_NO = PDI.STRIPNO
	AND ACT.SEG_NO = SUB.SEG_NO
	AND ACT.STRIP_NO = PDI.STRIPNO
	AND CF.STRIP_NO = PDI.STRIPNO
	AND PF.STRIP_NO = PDI.STRIPNO
	AND PF.SEG_NO = SUB.SEG_NO
	AND CG.STRIP_NO = PDI.STRIPNO
	AND PG.STRIP_NO = PDI.STRIPNO
	AND PG.SEG_NO = SUB.SEG_NO
	AND SF.STRIP_NO = PDI.STRIPNO
	AND SUB.STAND_NO = PF.STAND_NO
	AND CF.STAND_NO = PF.STAND_NO
	AND PG.STAND_NO = PF.STAND_NO
	AND CG.STAND_NO = PF.STAND_NO
	AND PDI.STRIPNO = RM.STRIP_NO
	AND RM.SEG_NO = SUB.SEG_NO
	--  AND PS.STRIPNO = PDI.STRIPNO
	AND CT.STRIP_NO = CF.STRIP_NO
	AND CT.STAND_NO = CF.STAND_NO
	AND SUB.SEG_NO = PT.SEG_NO
	AND SUB.STAND_NO = PT.STAND_NO
	AND SUB.STRIP_NO = PT.STRIP_NO
	AND PDI.STRIPNO = 'INPUT%%INPUT'
ORDER BY
	ACT.TOM,
	CF.STAND_NO
-- FETCH FIRST 1 ROWS ONLY
