# 差距审计：模型"答错"的下一步逐条清单（供化学家复核）

每条给出当前物种、GT 下一步、模型答案。自动类别基于 σ 图；初判由 Claude 给出（非化学家），acceptable = 约定/粒度差异或文献可辩护的替代，questionable = 无证据难以区分，wrong = 化学上不如 GT。

## A056 第 4/6 步 — 初判 **wrong**（chemistry；自动类别 partial；来源 one_sample）
- 条件：reagent: aq H2O2; reagent: NaOH; solvent: H2O; temperature: 50 °C
- 当前物种：`O=C1C=CC=CC12OC2[O-], O, [H+], [H+], [OH-], [OH-]`
- GT 下一步：`form C2-O9; form C8-O1; break C2-O1; break C3-C8; break C8-O9; C2-C3 order 1->1.5; C2-C7 order 1->1.5; C3-C4 order 1->1.5; C4-C5 order 2->1.5; C5-C6 order 1->1.5; C6-C7 order 2->1.5` → `O=COc1ccccc1[O-]`
- 模型答案：`form C2-O9; C2-O1 order 2->1; charge O1 +0->-1; charge O9 -1->+0` → `[O-]C12C=CC=CC13OC3O2`
- GT 后续：`form C8-O13; break C8-O11; charge O11 +0->-1; charge O13 -1->+0` ‖ `form H900010-O11; form H900011-O9; charge H900010 +1->+0; charge H900011 +1->+0; charge O11 -1->+0; charge O9 -1->+0`
- 初判理由：C2-O9 成环得三环过氧缩醛，不合理
- 化学家意见：

## B015 第 3/3 步 — 初判 **wrong**（chemistry；自动类别 partial；来源 select_own, one_sample）
- 条件：reagent: aq HBr; solvent: H2O; temperature: 0 °C
- 当前物种：`[Br-], [CH+](C1CCCCC1)C1CC1, O`
- GT 下一步：`form Br12-C11; break C11-C9; C2-C9 order 1->2; charge Br12 -1->+0; charge C2 +1->+0` → `BrCCC=CC1CCCCC1`
- 模型答案：`break C11-C9; C2-C9 order 1->2; charge C11 +0->+1; charge C2 +1->+0` → `[CH2+]CC=CC1CCCCC1`
- GT 后续：（无）
- 初判理由：环丙甲基正离子开环成一级碳正离子 [CH2+]，不如 GT 的 Br- 协同开环
- 化学家意见：

## C033 第 2/8 步 — 初判 **wrong**（chemistry；自动类别 subpath；来源 select_own, one_sample）
- 条件：reagent: n-BuLi; solvent: THF; temperature: -78 °C; reagent: PhSCl; temperature: -50 to 0 °C; solvent: THF, -78 °C; reagent: (0.4 eq) aq NH4Cl; temperature: rt
- 当前物种：`C=CC([O-])C=C, CCCC, ClSc1ccccc1, [CH2-]CCC, CCC1OC2OC(C)(C)OC(C=O)C2(C)C1(C)O, [NH4+]`
- GT 下一步：`form O4-S11; break Cl18-S11; charge Cl18 +0->-1; charge O4 -1->+0` → `C=CC(C=C)OSc1ccccc1, [Cl-]`
- 模型答案：`form C19-H900008; break C6-H900008; charge C19 -1->+0; charge C6 +0->-1` → `[CH-]=CC([O-])C=C, CCCC`
- GT 后续：`form C6-S11; break C3-O4; C3-C5 order 1->2; C5-C6 order 2->1; charge O4 +0->-1; charge S11 +0->+1` ‖ `form C19-H900008; break C6-H900008; charge C19 -1->+0; charge C6 +0->-1`
- 初判理由：BuLi 先夺烯基 C-H（不酸）；GT 的去质子发生在亚砜 α 位、[2,3] 之后
- 化学家意见：

## C060 第 3/5 步 — 初判 **wrong**（chemistry；自动类别 on_atoms；来源 select_own, one_sample）
- 条件：reagent: Pd2(dba)3-CHCl3 (2.5 mol%); reagent: DPPE; reagent: MECOCH2CO2Me; solvent: THF; temperature: rt; reagent: silica gel
- 当前物种：`CC(=C=[CH][Pd+])CO, COC(=O)[CH-]C(C)=O`
- GT 下一步：`form C12-C3; form C4-H900010; break C12-H900010; C12-C13 order 1->2; C13-O15 order 2->1; C3-C4 order 2->1; charge C12 -1->+0; charge O15 +0->-1` → `COC(=O)C(=C(C)[O-])C([CH2][Pd+])=C(C)CO`
- 模型答案：`form C12-Pd7; charge C12 -1->+0; charge Pd7 +1->+0` → `COC(=O)[CH]([Pd][CH]=C=C(C)CO)C(C)=O`
- GT 后续：`form C2-O15; break C4-Pd7; C2-C3 order 2->1; C3-C4 order 1->2; charge O15 -1->+0; charge Pd7 +1->+0` ‖ `form C4-H900011; break C2-C5; break H900011-O6; C12-C13 order 2->1.5; C12-C3 order 1->1.5; C13-O15 order 1->1.5; C2-C3 order 1->1.5; C2-O15 order 1->1.5; C3-C4 order 2->1; C5-O6 order 1->2`
- 初判理由：软亲核试剂（丙二酸酯类）进攻 Pd（内球）；已知为外球进攻中心碳
- 化学家意见：

## C066 第 3/8 步 — 初判 **wrong**（chemistry；自动类别 subpath；来源 select_own）
- 条件：reagent: C=C[Mg]Br; solvent: THF; temperature: -40 °C
- 当前物种：`C=[CH][Mg][Br], C=[CH][Mg][Br], C=[CH][Mg][Br], C=C[O][Mg][Br], Cc1ccccc1N=O, [H+]`
- GT 下一步：`form C16-O5; form Mg17-N6; break C16-Mg17; N6-O5 order 2->1` → `C=CO[N]([Mg][Br])c1ccccc1C`
- 模型答案：`form C16-N6; form Mg17-O5; break C16-Mg17; N6-O5 order 2->1` → `C=CN([O][Mg][Br])c1ccccc1C`
- GT 后续：`form C15-C9; break N6-O5; C10-C11 order 1.5->2; C10-C9 order 1.5->1; C11-C12 order 1.5->1; C12-C13 order 1.5->2; C13-C8 order 1.5->1; C15-C16 order 2->1; C16-O5 order 1->2; C8-C9 order 1.5->1; C8-N6 order 1->2` ‖ `form C16-N6; form Mg17-O5; break Mg17-N6; C16-O5 order 2->1`
- 初判理由：Bartoli：第二当量乙烯基格氏试剂进攻亚硝基 O（GT 正确），模型进攻 N
- 化学家意见：

## C066 第 3/8 步 — 初判 **wrong**（chemistry；自动类别 subpath；来源 one_sample）
- 条件：reagent: C=C[Mg]Br; solvent: THF; temperature: -40 °C
- 当前物种：`C=[CH][Mg][Br], C=[CH][Mg][Br], C=[CH][Mg][Br], C=C[O][Mg][Br], Cc1ccccc1N=O, [H+]`
- GT 下一步：`form C16-O5; form Mg17-N6; break C16-Mg17; N6-O5 order 2->1` → `C=CO[N]([Mg][Br])c1ccccc1C`
- 模型答案：`form C16-N6; break C16-Mg17; N6-O5 order 2->1; charge Mg17 +0->+1; charge O5 +0->-1` → `[Mg+][Br], C=CN([O-])c1ccccc1C`
- GT 后续：`form C15-C9; break N6-O5; C10-C11 order 1.5->2; C10-C9 order 1.5->1; C11-C12 order 1.5->1; C12-C13 order 1.5->2; C13-C8 order 1.5->1; C15-C16 order 2->1; C16-O5 order 1->2; C8-C9 order 1.5->1; C8-N6 order 1->2` ‖ `form C16-N6; form Mg17-O5; break Mg17-N6; C16-O5 order 2->1`
- 初判理由：同上，N-进攻
- 化学家意见：

## C106 第 2/6 步 — 初判 **wrong**（chemistry；自动类别 off_path；来源 select_own）
- 条件：solvent: toluene; temperature: 110 °C
- 当前物种：`C=Cc1ccccc1, [N-]=[N+]=CC1(CCC=CCOCC=Cc2ccccc2)CO1`
- GT 下一步：`break C5-O22; C4-C5 order 1->2; C4-N3 order 2->1; N2-N3 order 2->3; charge N2 -1->+0; charge O22 +0->-1` → `N#[N+]C=C(C[O-])CCC=CCOCC=Cc1ccccc1`
- 模型答案：`form C4-C8; form C9-N2; C4-N3 order 2->1; C8-C9 order 2->1; charge N2 -1->+0; charge N3 +1->+0` → `C(=Cc1ccccc1)COCC1N=NC2C1CCC21CO1`
- GT 后续：`form H900003-O22; break C4-H900003; break C4-N3; charge N3 +1->+0; charge O22 -1->+0` ‖ `form C4-C8; form C4-C9; C8-C9 order 2->1`
- 初判理由：分子内重氮 1,3-偶极环加成：本身合理的竞争反应，但不是本文路线
- 化学家意见：

## A066 第 2/3 步 — 初判 **questionable**（order；自动类别 subpath；来源 select_own）
- 条件：temperature: 630 °C; pressure: 12 mmHg
- 当前物种：`C=c1ccccc1=C(O)Cl`
- GT 下一步：`form Cl10-H900003; break C8-Cl10; break H900003-O9; C8-O9 order 1->2` → `C=c1ccccc1=C=O, Cl`
- 模型答案：`form C1-C8; C1-C2 order 2->1; C7-C8 order 2->1` → `OC1(Cl)Cc2ccccc21`
- GT 后续：`form C1-C8; C1-C2 order 2->1; C7-C8 order 2->1`
- 初判理由：630 °C 下先电环化成苯并环丁烯醇、后失 HCl，与 GT 顺序相反；无实验证据难以区分
- 化学家意见：

## C106 第 2/6 步 — 初判 **questionable**（order；自动类别 subpath；来源 one_sample）
- 条件：solvent: toluene; temperature: 110 °C
- 当前物种：`C=Cc1ccccc1, [N-]=[N+]=CC1(CCC=CCOCC=Cc2ccccc2)CO1`
- GT 下一步：`break C5-O22; C4-C5 order 1->2; C4-N3 order 2->1; N2-N3 order 2->3; charge N2 -1->+0; charge O22 +0->-1` → `N#[N+]C=C(C[O-])CCC=CCOCC=Cc1ccccc1`
- 模型答案：`break C4-N3; N2-N3 order 2->3; charge N2 -1->+0; charge N3 +1->+0` → `[CH]C1(CCC=CCOCC=Cc2ccccc2)CO1, N#N`
- GT 后续：`form H900003-O22; break C4-H900003; break C4-N3; charge N3 +1->+0; charge O22 -1->+0` ‖ `form C4-C8; form C4-C9; C8-C9 order 2->1`
- 初判理由：先失 N2 成卡宾、再开环氧，与 GT（先开环成乙烯基重氮盐）顺序相反
- 化学家意见：

## A031 第 6/7 步 — 初判 **acceptable**（metric artifact；自动类别 same_molecules；来源 select_own, one_sample）
- 条件：reagent: Hg(OAc)2; solvent: Et2O-H2O; temperature: rt; reagent: NaBH4; solvent: aq NaOH
- 当前物种：`CC(=O)[O-], CC1(O)CCCC[CH]1[Hg], CC(=O)O, C, CC1(O)CCCC[CH]1[HgH], CC(=O)[O-]`
- GT 下一步：`form C10-H900012; break C10-Hg5; break C13-H900012` → `CC1(O)[CH]CCCC1, [Hg]`
- 模型答案：`break C10-Hg5` → `CC1(O)[CH]CCCC1, [Hg]`
- GT 后续：`form C13-H900041; break H900041-Hg31`
- 初判理由：C-Hg 均裂得烷基自由基：SMILES 与 GT 下一态相同，仅原子映射不同
- 化学家意见：

## A056 第 3/6 步 — 初判 **acceptable**（metric artifact + literature；自动类别 on_atoms；来源 select_own, one_sample）
- 条件：reagent: aq H2O2; reagent: NaOH; solvent: H2O; temperature: 50 °C
- 当前物种：`[O-]c1ccccc1C([O-])OO, O, [H+], [H+], [OH-]`
- GT 下一步：`form C2-C7; form C3-O11; form C8-H900001; form C8-O11; break C2-H900001; break C2-O11; break C7-C8; break O11-O12; C2-O1 order 1->2; C3-C4 order 1.5->1; C3-C8 order 1.5->1; C4-C5 order 1.5->2; C5-C6 order 1.5->1; C6-C7 order 1.5->2; charge O1 -1->+0; charge O12 +0->-1` → `O=C1C=CC=CC12OC2[O-], [OH-]`
- 模型答案：`form C3-O11; break C2-C3; break O11-O12; C2-O1 order 1->2; charge O1 -1->+0; charge O12 +0->-1` → `O=COc1ccccc1[O-], [OH-]`
- GT 后续：`form C2-O9; form C8-O1; break C2-O1; break C3-C8; break C8-O9; C2-C3 order 1->1.5; C2-C7 order 1->1.5; C3-C4 order 1->1.5; C4-C5 order 2->1.5; C5-C6 order 1->1.5; C6-C7 order 2->1.5` ‖ `form C8-O13; break C8-O11; charge O11 +0->-1; charge O13 -1->+0`
- 初判理由：教科书 Dakin：芳基 C→O 迁移直接得甲酸芳酯，与 GT 第 4 态同一分子（映射不同）；GT 经螺环中间体的路线反而少见
- 化学家意见：

## A056 第 4/6 步 — 初判 **acceptable**（metric artifact；自动类别 same_molecules；来源 select_own）
- 条件：reagent: aq H2O2; reagent: NaOH; solvent: H2O; temperature: 50 °C
- 当前物种：`O=C1C=CC=CC12OC2[O-], O, [H+], [H+], [OH-], [OH-]`
- GT 下一步：`form C2-O9; form C8-O1; break C2-O1; break C3-C8; break C8-O9; C2-C3 order 1->1.5; C2-C7 order 1->1.5; C3-C4 order 1->1.5; C4-C5 order 2->1.5; C5-C6 order 1->1.5; C6-C7 order 2->1.5` → `O=COc1ccccc1[O-]`
- 模型答案：`break C3-C8; C2-C3 order 1->1.5; C2-C7 order 1->1.5; C2-O1 order 2->1; C3-C4 order 1->1.5; C4-C5 order 2->1.5; C5-C6 order 1->1.5; C6-C7 order 2->1.5; C8-O9 order 1->2; charge O1 +0->-1; charge O9 -1->+0` → `O=COc1ccccc1[O-]`
- GT 后续：`form C8-O13; break C8-O11; charge O11 +0->-1; charge O13 -1->+0` ‖ `form H900010-O11; form H900011-O9; charge H900010 +1->+0; charge H900011 +1->+0; charge O11 -1->+0; charge O9 -1->+0`
- 初判理由：产物 SMILES 与 GT 下一态相同（映射不同）
- 化学家意见：

## A056 第 5/6 步 — 初判 **acceptable**（granularity；自动类别 partial；来源 select_own, one_sample）
- 条件：reagent: aq H2O2; reagent: NaOH; solvent: H2O; temperature: 50 °C
- 当前物种：`O=COc1ccccc1[O-], O, [H+], [H+], [OH-], [OH-]`
- GT 下一步：`form C8-O13; break C8-O11; charge O11 +0->-1; charge O13 -1->+0` → `[O-]c1ccccc1[O-], O=CO`
- 模型答案：`form C8-O13; C8-O1 order 2->1; charge O1 +0->-1; charge O13 -1->+0` → `[O-]c1ccccc1OC([O-])O`
- GT 后续：`form H900010-O11; form H900011-O9; charge H900010 +1->+0; charge H900011 +1->+0; charge O11 -1->+0; charge O9 -1->+0`
- 初判理由：甲酸酯碱水解经四面体中间体（BAc2 分步），GT 为协同
- 化学家意见：

## A057 第 5/8 步 — 初判 **acceptable**（proton/tautomer；自动类别 partial；来源 select_own, one_sample）
- 条件：reagent: NaOCl; reagent: NaOH; solvent: H2O; temperature: 50 °C
- 当前物种：`COc1ccc(N=C=O)cc1OC, [Cl-], O, O, [OH-], O`
- GT 下一步：`form C7-O17; form H900014-O9; break H900014-O17; C7-O9 order 2->1` → `COc1ccc(N=C([O-])O)cc1OC`
- 模型答案：`form C7-O17; C7-N8 order 2->1; charge N8 +0->-1; charge O17 -1->+0` → `COc1ccc([N-]C(=O)O)cc1OC`
- GT 后续：`form H900016-N8; break H900016-O18; C7-N8 order 2->1; C7-O17 order 1->2; charge O17 -1->+0; charge O18 +0->-1` ‖ `form H900014-O18; break H900014-O9; charge O18 -1->+0; charge O9 +0->-1`
- 初判理由：OH- 加成异氰酸酯得 N 负离子；与 GT 只差 H 在两个 O 之间（氨基甲酸根互变），GT 把质子转移捆进了加成
- 化学家意见：

## A057 第 6/8 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own）
- 条件：reagent: NaOCl; reagent: NaOH; solvent: H2O; temperature: 50 °C
- 当前物种：`COc1ccc(N=C([O-])O)cc1OC, [Cl-], O, O, O`
- GT 下一步：`form H900016-N8; break H900016-O18; C7-N8 order 2->1; C7-O17 order 1->2; charge O17 -1->+0; charge O18 +0->-1` → `COc1ccc(NC(=O)O)cc1OC, [OH-]`
- 模型答案：`form H900014-O18; form H900015-N8; break H900014-O9; break H900015-O18; C7-N8 order 2->1; C7-O9 order 1->2` → `COc1ccc(NC(=O)[O-])cc1OC, O`
- GT 后续：`form H900014-O18; break H900014-O9; charge O18 -1->+0; charge O9 +0->-1` ‖ `form H900015-N8; break C7-N8; break H900015-O18; C7-O9 order 1->2; charge O18 +0->-1; charge O9 -1->+0`
- 初判理由：两次质子转移合并
- 化学家意见：

## A057 第 7/8 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own）
- 条件：reagent: NaOCl; reagent: NaOH; solvent: H2O; temperature: 50 °C
- 当前物种：`COc1ccc(NC(=O)O)cc1OC, [Cl-], O, O, [OH-]`
- GT 下一步：`form H900014-O18; break H900014-O9; charge O18 -1->+0; charge O9 +0->-1` → `COc1ccc(NC(=O)[O-])cc1OC, O`
- 模型答案：`form H900014-O18; form H900015-N8; break C7-N8; break H900014-O9; break H900015-O18; C7-O9 order 1->2` → `COc1ccc(N)cc1OC, O=C=O, [OH-]`
- GT 后续：`form H900015-N8; break C7-N8; break H900015-O18; C7-O9 order 1->2; charge O18 +0->-1; charge O9 -1->+0`
- 初判理由：质子转移与脱羧合并
- 化学家意见：

## A059 第 1/4 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：reagent: hv; solvent: dioxane; solvent: Et2O-H2O
- 当前物种：`CC12CCC3C(CC=C4CC(O)CCC43C)C1CC(=[N+]=[N-])C2=O, O`
- GT 下一步：`break C19-N20; N20-N21 order 2->3; charge N20 +1->+0; charge N21 -1->+0` → `CC12CCC3C(CC=C4CC(O)CCC43C)C1C[C]C2=O, N#N`
- 模型答案：`form C15-C19; break C15-C17; break C19-N20; C17-C19 order 1->2; N20-N21 order 2->3; charge N20 +1->+0; charge N21 -1->+0` → `CC12CCC3C(CC=C4CC(O)CCC43C)C1CC2=C=O, N#N`
- GT 后续：`form C15-C19; break C15-C17; C17-C19 order 1->2` ‖ `form C17-O24; form H900028-O18; break H900028-O24; C17-O18 order 2->1`
- 初判理由：协同 Wolff 重排（失 N2 与 1,2-迁移合一）；光化学 Wolff 是否经游离卡宾本身有争议
- 化学家意见：

## A059 第 3/4 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：reagent: hv; solvent: dioxane; solvent: Et2O-H2O
- 当前物种：`CC12CCC3C(CC=C4CC(O)CCC43C)C1CC2=C=O, O, N#N`
- GT 下一步：`form C17-O24; form H900028-O18; break H900028-O24; C17-O18 order 2->1` → `CC12CCC(O)CC1=CCC1C2CCC2(C)C(=C(O)O)CC12`
- 模型答案：`form C17-O24; form C19-H900028; break H900028-O24; C17-C19 order 2->1` → `CC12CCC(O)CC1=CCC1C2CCC2(C)C(C(=O)O)CC12`
- GT 后续：`form C19-H900028; break H900028-O18; C17-C19 order 2->1; C17-O18 order 1->2`
- 初判理由：烯酮水合直接到羧酸，跳过 1,1-烯二醇
- 化学家意见：

## A060 第 4/4 步 — 初判 **acceptable**（granularity；自动类别 partial；来源 one_sample）
- 条件：reagent: NaOMe; solvent: Et2O; temperature: reflux
- 当前物种：`CO, COC1([O-])C2CCCC21, CO, [Cl-]`
- GT 下一步：`form C5-H900019; break C4-C5; break H900019-O14; C4-O3 order 1->2; charge O14 +0->-1; charge O3 -1->+0` → `COC(=O)C1CCCC1, C[O-]`
- 模型答案：`break C4-C5; C4-O3 order 1->2; charge C5 +0->-1; charge O3 -1->+0` → `COC(=O)C1[CH-]CCC1`
- GT 后续：（无）
- 初判理由：Favorskii 环开裂得碳负离子再质子化（教科书分步），GT 与质子化协同
- 化学家意见：

## A066 第 1/3 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：temperature: 630 °C; pressure: 12 mmHg
- 当前物种：`Cc1ccccc1C(=O)Cl`
- GT 下一步：`form H900003-O9; break C1-H900003; C1-C2 order 1->2; C7-C8 order 1->2; C8-O9 order 2->1` → `C=c1ccccc1=C(O)Cl`
- 模型答案：`form Cl10-H900003; break C1-H900003; break C8-Cl10; C1-C2 order 1->2; C2-C3 order 1.5->1; C2-C7 order 1.5->1; C3-C4 order 1.5->2; C4-C5 order 1.5->1; C5-C6 order 1.5->2; C6-C7 order 1.5->1; C7-C8 order 1->2` → `C=c1ccccc1=C=O, Cl`
- GT 后续：`form Cl10-H900003; break C8-Cl10; break H900003-O9; C8-O9 order 1->2` ‖ `form C1-C8; C1-C2 order 2->1; C7-C8 order 2->1`
- 初判理由：FVP：1,5-H 迁移与 HCl 消除合并
- 化学家意见：

## B024 第 1/4 步 — 初判 **acceptable**（granularity；自动类别 partial；来源 select_own, one_sample）
- 条件：reagent: aq KOH; solvent: EtOH; temperature: 10 to 20 °C
- 当前物种：`Cc1ccc(S(=O)(=O)N(C)N=O)cc1, O=C1CCCCC1, [OH-], [OH-]`
- GT 下一步：`form H900011-O4; form O15-S5; break H900011-O15; break N2-S5; N2-N3 order 1->2; N3-O4 order 2->1; O14-S5 order 2->1; charge O14 +0->-1; charge O15 -1->+0` → `CN=NO, Cc1ccc(S(=O)(=O)[O-])cc1`
- 模型答案：`form O15-S5; O14-S5 order 2->1; charge O14 +0->-1; charge O15 -1->+0` → `Cc1ccc(S(=O)([O-])(O)N(C)N=O)cc1`
- GT 后续：`form H900003-O16; break C1-H900003; break N3-O4; N2-N3 order 2->3; charge C1 +0->-1; charge N2 +0->+1; charge O16 -1->+0; charge O4 +0->-1` ‖ `form C1-C17; C17-O18 order 2->1; charge C1 -1->+0; charge O18 +0->-1`
- 初判理由：OH- 进攻磺酰 S 形成五配位中间体（加成-消除），GT 为协同；文献两种画法都有
- 化学家意见：

## B029_b 第 1/4 步 — 初判 **acceptable**（granularity；自动类别 partial；来源 select_own, one_sample）
- 条件：reagent: HCHO; solvent: toluene; temperature: 0 °C to rt; temperature: 110 °C
- 当前物种：`C=CCCCNO, C=O`
- GT 下一步：`form C8-N6; form H900010-O9; break H900010-N6; C8-O9 order 2->1` → `C=CCCCN(O)CO`
- 模型答案：`form C8-N6; C8-O9 order 2->1; charge N6 +0->+1; charge O9 +0->-1` → `C=CCCC[NH+](O)C[O-]`
- GT 后续：`break C8-O9; C8-N6 order 1->2; charge N6 +0->+1; charge O9 +0->-1` ‖ `form H900011-O9; break H900011-O7; charge O7 +0->-1; charge O9 -1->+0`
- 初判理由：羟胺加成甲醛得两性离子四面体中间体，GT 把质子转移捆进加成
- 化学家意见：

## B029_b 第 2/4 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：reagent: HCHO; solvent: toluene; temperature: 0 °C to rt; temperature: 110 °C
- 当前物种：`C=CCCCN(O)CO`
- GT 下一步：`break C8-O9; C8-N6 order 1->2; charge N6 +0->+1; charge O9 +0->-1` → `C=CCCC[N+](=C)O, [OH-]`
- 模型答案：`form H900011-O9; break C8-O9; break H900011-O7; C8-N6 order 1->2; charge N6 +0->+1; charge O7 +0->-1` → `C=CCCC[N+](=C)[O-], O`
- GT 后续：`form H900011-O9; break H900011-O7; charge O7 +0->-1; charge O9 -1->+0` ‖ `form C1-C8; form C2-O7; C1-C2 order 2->1; C8-N6 order 2->1; charge N6 +1->+0; charge O7 -1->+0`
- 初判理由：失水与质子转移合并成硝酮
- 化学家意见：

## B032 第 1/4 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：reagent: hv; solvent: MeOH
- 当前物种：`O=C1C=CCCC1, CO, O=C1C=CCCC1, CO`
- GT 下一步：`C2-O1 order 2->1` → `[O][C]1C=CCCC1`
- 模型答案：`form H900011-O1; break C8-H900011; C2-O1 order 2->1` → `O[C]1C=CCCC1, [CH2]O`
- GT 后续：`form H900011-O1; break C8-H900011` ‖ `form C13-C8; C12-C13 order 2->1`
- 初判理由：n,π* 激发与夺氢合并
- 化学家意见：

## B096 第 5/7 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：reagent: C=CCBr; reagent: K2CO3; solvent: DMF; temperature: 20 °C; reagent: CNCC(OCC)=O; solvent: toluene; temperature: reflux; reagent: Dean-Stark trap
- 当前物种：`[Br-], C=CCOc1ccccc1C(O)N(C)CC(=O)OCC, O`
- GT 下一步：`break C2-O1; C2-N15 order 1->2; charge N15 +0->+1; charge O1 +0->-1` → `C=CCOc1ccccc1C=[N+](C)CC(=O)OCC, [OH-]`
- 模型答案：`form H900018-O1; break C17-H900018; break C2-O1; C2-N15 order 1->2; charge C17 +0->-1; charge N15 +0->+1` → `C=CCOc1ccccc1C=[N+](C)[CH-]C(=O)OCC, O`
- GT 后续：`form H900018-O1; break C17-H900018; charge C17 +0->-1; charge O1 -1->+0` ‖ `form C11-C17; form C12-C2; C11-C12 order 2->1; C2-N15 order 2->1; charge C17 -1->+0; charge N15 +1->+0`
- 初判理由：亚胺正离子形成与去质子成甲亚胺叶立德合并
- 化学家意见：

## B099 第 5/5 步 — 初判 **acceptable**（granularity；自动类别 partial；来源 select_own）
- 条件：reagent: MeP(OEt)2; reagent: n-BuLi; solvent: THF; temperature: -78 °C
- 当前物种：`CCOP(=O)(OCC)C1C(=O)CCC1(C)[O-]`
- GT 下一步：`form O7-P9; break C2-O7; break C8-P9; C2-C8 order 1->2` → `CC1=CC(=O)CC1, CCOP(=O)([O-])OCC`
- 模型答案：`form O7-P9; O10-P9 order 2->1; charge O10 +0->-1; charge O7 -1->+0` → `CCOP1([O-])(OCC)OC2(C)CCC(=O)C21`
- GT 后续：（无）
- 初判理由：形成氧杂磷杂环丁烷（五配位 P）中间体，教科书 HWE 画法；GT 协同
- 化学家意见：

## C033 第 3/8 步 — 初判 **acceptable**（metric artifact；自动类别 same_molecules；来源 select_own, one_sample）
- 条件：reagent: n-BuLi; solvent: THF; temperature: -78 °C; reagent: PhSCl; temperature: -50 to 0 °C; solvent: THF, -78 °C; reagent: (0.4 eq) aq NH4Cl; temperature: rt
- 当前物种：`C=CC(C=C)OSc1ccccc1, CCCC, [CH2-]CCC, CCC1OC2OC(C)(C)OC(C=O)C2(C)C1(C)O, [Cl-], [NH4+]`
- GT 下一步：`form C6-S11; break C3-O4; C3-C5 order 1->2; C5-C6 order 2->1; charge O4 +0->-1; charge S11 +0->+1` → `C=CC=CC[S+]([O-])c1ccccc1`
- 模型答案：`form C1-S11; break C3-O4; C1-C2 order 2->1; C2-C3 order 1->2; charge O4 +0->-1; charge S11 +0->+1` → `C=CC=CC[S+]([O-])c1ccccc1`
- GT 后续：`form C19-H900008; break C6-H900008; charge C19 -1->+0; charge C6 +0->-1` ‖ `form C23-C6; C23-O24 order 2->1; charge C6 -1->+0; charge O24 +0->-1`
- 初判理由：对称二乙烯基甲醇：经 C1 或 C6 的 [2,3] 得到同一分子，按映射判错
- 化学家意见：

## C033 第 7/8 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：reagent: n-BuLi; solvent: THF; temperature: -78 °C; reagent: PhSCl; temperature: -50 to 0 °C; solvent: THF, -78 °C; reagent: (0.4 eq) aq NH4Cl; temperature: rt
- 当前物种：`C=CC=CC(C(O)C1OC(C)(C)OC2OC(CC)C(C)(O)C21C)[S+]([O-])c1ccccc1, CCCC, CCCC, [Cl-], N`
- GT 下一步：`form C3-O4; break C6-S11; C3-C5 order 2->1; C5-C6 order 1->2; charge O4 -1->+0; charge S11 +1->+0` → `C=CC(C=CC(O)C1OC(C)(C)OC2OC(CC)C(C)(O)C21C)OSc1ccccc1`
- 模型答案：`form C1-S11; break C6-S11; C1-C2 order 2->1; C2-C3 order 1->2; C3-C5 order 2->1; C5-C6 order 1->2` → `CCC1OC2OC(C)(C)OC(C(O)C=CC=CC[S+]([O-])c3ccccc3)C2(C)C1(C)O`
- GT 后续：`form C1-S11; break C3-O4; C1-C2 order 2->1; C2-C3 order 1->2; charge O4 +0->-1; charge S11 +0->+1`
- 初判理由：两次 [2,3] 合并
- 化学家意见：

## C046 第 1/4 步 — 初判 **acceptable**（literature；自动类别 partial；来源 select_own）
- 条件：solvent: toluene; temperature: reflux
- 当前物种：`C=CC(=CCCCOC1(OC)N=NC(C)(C)O1)C(=O)OC`
- GT 下一步：`break C13-N16; break C9-N17; break C9-O12; C13-O12 order 1->2; N16-N17 order 2->3` → `C=CC(=CCCCO[C]OC)C(=O)OC, CC(C)=O, N#N`
- 模型答案：`break C13-N16; break C9-N17; C13-O12 order 1->2; N16-N17 order 2->3; charge C9 +0->-1; charge O12 +0->+1` → `C=CC(=CCCCO[C-](OC)[O+]=C(C)C)C(=O)OC, N#N`
- GT 后续：`form C3-C9; form C4-C9; C3-C4 order 2->1` ‖ `break C3-C9; C18-C3 order 1->2; C18-O19 order 2->1; C9-O8 order 1->2; charge O19 +0->-1; charge O8 +0->+1`
- 初判理由：噁二唑啉热解先失 N2 得羰基叶立德（Warkentin 机理），GT 直接到二烷氧基卡宾
- 化学家意见：

## C046 第 2/4 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：solvent: toluene; temperature: reflux
- 当前物种：`C=CC(=CCCCO[C]OC)C(=O)OC, CC(C)=O, N#N`
- GT 下一步：`form C3-C9; form C4-C9; C3-C4 order 2->1` → `C=CC1(C(=O)OC)C2CCCOC21OC`
- 模型答案：`form C1-C9; form C4-C9; C1-C2 order 2->1; C2-C3 order 1->2; C3-C4 order 2->1` → `COC(=O)C1=CCC2(OC)OCCCC12`
- GT 后续：`break C3-C9; C18-C3 order 1->2; C18-O19 order 2->1; C9-O8 order 1->2; charge O19 +0->-1; charge O8 +0->+1` ‖ `form C1-C9; C18-C3 order 2->1; C18-O19 order 1->2; C1-C2 order 2->1; C2-C3 order 1->2; C9-O8 order 2->1; charge O19 -1->+0; charge O8 +1->+0`
- 初判理由：形式 [4+1] 一步，跳过环丙烷中间体（GT 经乙烯基环丙烷重排）
- 化学家意见：

## C046 第 3/4 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：solvent: toluene; temperature: reflux
- 当前物种：`C=CC1(C(=O)OC)C2CCCOC21OC, CC(C)=O, N#N`
- GT 下一步：`break C3-C9; C18-C3 order 1->2; C18-O19 order 2->1; C9-O8 order 1->2; charge O19 +0->-1; charge O8 +0->+1` → `C=CC(=C([O-])OC)C1CCC[O+]=C1OC`
- 模型答案：`form C1-C9; break C3-C9; C1-C2 order 2->1; C2-C3 order 1->2` → `COC(=O)C1=CCC2(OC)OCCCC12`
- GT 后续：`form C1-C9; C18-C3 order 2->1; C18-O19 order 1->2; C1-C2 order 2->1; C2-C3 order 1->2; C9-O8 order 2->1; charge O19 -1->+0; charge O8 +1->+0`
- 初判理由：开环与再闭环合并
- 化学家意见：

## C060 第 4/5 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own）
- 条件：reagent: Pd2(dba)3-CHCl3 (2.5 mol%); reagent: DPPE; reagent: MECOCH2CO2Me; solvent: THF; temperature: rt; reagent: silica gel
- 当前物种：`COC(=O)C(=C(C)[O-])C([CH2][Pd+])=C(C)CO`
- GT 下一步：`form C2-O15; break C4-Pd7; C2-C3 order 2->1; C3-C4 order 1->2; charge O15 -1->+0; charge Pd7 +1->+0` → `C=C1C(C(=O)OC)=C(C)OC1(C)CO, [Pd]`
- 模型答案：`form C2-O15; form C4-H900011; break C2-C5; break C4-Pd7; break H900011-O6; C5-O6 order 1->2; charge O15 -1->+0; charge Pd7 +1->+0` → `COC(=O)c1c(C)oc(C)c1C, C=O, [Pd]`
- GT 后续：`form C4-H900011; break C2-C5; break H900011-O6; C12-C13 order 2->1.5; C12-C3 order 1->1.5; C13-O15 order 1->1.5; C2-C3 order 1->1.5; C2-O15 order 1->1.5; C3-C4 order 2->1; C5-O6 order 1->2`
- 初判理由：环化与后续步骤合并
- 化学家意见：

## C066 第 1/8 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：reagent: C=C[Mg]Br; solvent: THF; temperature: -40 °C
- 当前物种：`C=[CH][Mg][Br], C=[CH][Mg][Br], C=[CH][Mg][Br], C=[CH][Mg][Br], Cc1ccccc1[N+](=O)[O-], [H+]`
- GT 下一步：`form C2-O7; break C2-Mg3; N6-O7 order 2->1; charge Mg3 +0->+1; charge N6 +1->+0` → `[Mg+][Br], C=CON([O-])c1ccccc1C`
- 模型答案：`form C2-O7; form Mg3-O7; break C2-Mg3; break N6-O7; N6-O5 order 1->2; charge N6 +1->+0; charge O5 -1->+0` → `C=C[O][Mg][Br], Cc1ccccc1N=O`
- GT 后续：`form Mg3-O7; break N6-O7; N6-O5 order 1->2; charge Mg3 +1->+0; charge O5 -1->+0` ‖ `form C16-O5; form Mg17-N6; break C16-Mg17; N6-O5 order 2->1`
- 初判理由：Bartoli 第一步两段合并
- 化学家意见：

## C066 第 5/8 步 — 初判 **acceptable**（order；自动类别 subpath；来源 select_own, one_sample）
- 条件：reagent: C=C[Mg]Br; solvent: THF; temperature: -40 °C
- 当前物种：`CC1=CC=CC(CC=O)C1=[N][Mg][Br], C=[CH][Mg][Br], C=[CH][Mg][Br], C=C[O][Mg][Br], [H+]`
- GT 下一步：`form C16-N6; form Mg17-O5; break Mg17-N6; C16-O5 order 2->1` → `CC1=CC=CC2CC([O][Mg][Br])N=C12`
- 模型答案：`form C20-H900004; break Br22-Mg21; break C20-Mg21; break C9-H900004; C10-C11 order 2->1.5; C10-C9 order 1->1.5; C11-C12 order 1->1.5; C12-C13 order 2->1.5; C13-C8 order 1->1.5; C8-C9 order 1->1.5; C8-N6 order 2->1; charge Br22 +0->-1; charge Mg21 +0->+2; charge N6 +0->-1` → `Cc1cccc(CC=O)c1[N-][Mg][Br], [Br-], C=C, [Mg+2]`
- GT 后续：`form C20-H900004; break Br22-Mg21; break C20-Mg21; break C9-H900004; C10-C11 order 2->1.5; C10-C9 order 1->1.5; C11-C12 order 1->1.5; C12-C13 order 2->1.5; C13-C8 order 1->1.5; C8-C9 order 1->1.5; C8-N6 order 2->1; charge Br22 +0->-1; charge Mg21 +0->+2; charge N6 +0->-1` ‖ `form C24-H900012; form Mg25-N6; break C15-H900012; break C16-O5; break C24-Mg25; C15-C16 order 1->1.5; C15-C9 order 1->1.5; C16-N6 order 1->1.5; C8-N6 order 1->1.5; charge N6 -1->+0; charge O5 +0->-1`
- 初判理由：[3,3] 后先重芳构化、再环化：常见教科书 Bartoli 顺序，GT 先环化
- 化学家意见：

## C066 第 7/8 步 — 初判 **acceptable**（counterion；自动类别 subpath；来源 select_own）
- 条件：reagent: C=C[Mg]Br; solvent: THF; temperature: -40 °C
- 当前物种：`Cc1cccc2c1[N-]C([O][Mg][Br])C2, [Br-], C=[CH][Mg][Br], C=C[O][Mg][Br], C=C, [H+], [Mg+2]`
- GT 下一步：`form C24-H900012; form Mg25-N6; break C15-H900012; break C16-O5; break C24-Mg25; C15-C16 order 1->1.5; C15-C9 order 1->1.5; C16-N6 order 1->1.5; C8-N6 order 1->1.5; charge N6 -1->+0; charge O5 +0->-1` → `[O-][Mg][Br], Cc1cccc2cc[n]([Mg][Br])c12, C=C`
- 模型答案：`form C24-H900012; break Br26-Mg25; break C15-H900012; break C16-O5; break C24-Mg25; C15-C16 order 1->1.5; C15-C9 order 1->1.5; C16-N6 order 1->1.5; C8-N6 order 1->1.5; charge Br26 +0->-1; charge Mg25 +0->+2; charge O5 +0->-1` → `[O-][Mg][Br], [Br-], Cc1cccc2cc[n-]c12, C=C, [Mg+2]`
- GT 后续：`form H900020-N6; break Br26-Mg25; break Mg25-N6; charge Br26 +0->-1; charge H900020 +1->+0; charge Mg25 +0->+2`
- 初判理由：与 GT 同一事件，只差 N-MgBr 键与游离 MgBr2/N 负离子的记账
- 化学家意见：

## C066 第 7/8 步 — 初判 **acceptable**（granularity；自动类别 partial；来源 one_sample）
- 条件：reagent: C=C[Mg]Br; solvent: THF; temperature: -40 °C
- 当前物种：`Cc1cccc2c1[N-]C([O][Mg][Br])C2, [Br-], C=[CH][Mg][Br], C=C[O][Mg][Br], C=C, [H+], [Mg+2]`
- GT 下一步：`form C24-H900012; form Mg25-N6; break C15-H900012; break C16-O5; break C24-Mg25; C15-C16 order 1->1.5; C15-C9 order 1->1.5; C16-N6 order 1->1.5; C8-N6 order 1->1.5; charge N6 -1->+0; charge O5 +0->-1` → `[O-][Mg][Br], Cc1cccc2cc[n]([Mg][Br])c12, C=C`
- 模型答案：`break C16-O5; C16-N6 order 1->2; charge N6 -1->+0; charge O5 +0->-1` → `[O-][Mg][Br], Cc1cccc2c1N=CC2`
- GT 后续：`form H900020-N6; break Br26-Mg25; break Mg25-N6; charge Br26 +0->-1; charge H900020 +1->+0; charge Mg25 +0->+2`
- 初判理由：先消除成 3H-吲哚亚胺、再芳构化，GT 协同
- 化学家意见：

## C075 第 2/5 步 — 初判 **acceptable**（literature；自动类别 subpath；来源 select_own, one_sample）
- 条件：reagent: Rh2(OAc)4; solvent: benzene; temperature: reflux; reagent: BF3-Et2O; temperature: rt
- 当前物种：`COC(=O)[C](=[Rh])C(C)=O, CC=CC(C)O, N#N`
- GT 下一步：`form C5-O17; C5-Rh11 order 2->1; charge O17 +0->+1; charge Rh11 +0->-1` → `CC=CC(C)[OH+][C]([Rh-])(C(C)=O)C(=O)OC`
- 模型答案：`form C5-O17; break C5-Rh11; charge C5 +0->-1; charge O17 +0->+1` → `CC=CC(C)[OH+][C-](C(C)=O)C(=O)OC, [Rh]`
- GT 后续：`form H900016-O8; break C5-Rh11; break H900016-O17; C5-C6 order 1->2; C6-O8 order 2->1; charge O17 +1->+0; charge Rh11 -1->+0` ‖ `form C13-C6; break C15-O17; C13-C14 order 2->1; C14-C15 order 1->2; C5-C6 order 2->1; C5-O17 order 1->2`
- 初判理由：游离氧鎓叶立德 vs 金属结合叶立德，文献本身有争议
- 化学家意见：

## C075 第 3/5 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own）
- 条件：reagent: Rh2(OAc)4; solvent: benzene; temperature: reflux; reagent: BF3-Et2O; temperature: rt
- 当前物种：`CC=CC(C)[OH+][C]([Rh-])(C(C)=O)C(=O)OC, N#N`
- GT 下一步：`form H900016-O8; break C5-Rh11; break H900016-O17; C5-C6 order 1->2; C6-O8 order 2->1; charge O17 +1->+0; charge Rh11 -1->+0` → `CC=CC(C)OC(C(=O)OC)=C(C)O, [Rh]`
- 模型答案：`form C13-C5; break C15-O17; break C5-Rh11; C13-C14 order 2->1; C14-C15 order 1->2; charge O17 +1->+0; charge Rh11 -1->+0` → `CC=CC(C)C(O)(C(C)=O)C(=O)OC, [Rh]`
- GT 后续：`form C13-C6; break C15-O17; C13-C14 order 2->1; C14-C15 order 1->2; C5-C6 order 2->1; C5-O17 order 1->2` ‖ `form C13-C5; form H900016-O17; break C13-C6; break H900016-O8; C5-O17 order 2->1; C6-O8 order 1->2`
- 初判理由：合并了烯醇化与 [2,3]
- 化学家意见：

## C075 第 3/5 步 — 初判 **acceptable**（granularity；自动类别 partial；来源 one_sample）
- 条件：reagent: Rh2(OAc)4; solvent: benzene; temperature: reflux; reagent: BF3-Et2O; temperature: rt
- 当前物种：`CC=CC(C)[OH+][C]([Rh-])(C(C)=O)C(=O)OC, N#N`
- GT 下一步：`form H900016-O8; break C5-Rh11; break H900016-O17; C5-C6 order 1->2; C6-O8 order 2->1; charge O17 +1->+0; charge Rh11 -1->+0` → `CC=CC(C)OC(C(=O)OC)=C(C)O, [Rh]`
- 模型答案：`break C5-Rh11; charge C5 +0->-1; charge Rh11 -1->+0` → `CC=CC(C)[OH+][C-](C(C)=O)C(=O)OC, [Rh]`
- GT 后续：`form C13-C6; break C15-O17; C13-C14 order 2->1; C14-C15 order 1->2; C5-C6 order 2->1; C5-O17 order 1->2` ‖ `form C13-C5; form H900016-O17; break C13-C6; break H900016-O8; C5-O17 order 2->1; C6-O8 order 1->2`
- 初判理由：Rh 解离得游离叶立德，GT 与质子转移协同
- 化学家意见：

## C075 第 4/5 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：reagent: Rh2(OAc)4; solvent: benzene; temperature: reflux; reagent: BF3-Et2O; temperature: rt
- 当前物种：`CC=CC(C)OC(C(=O)OC)=C(C)O, N#N, [Rh]`
- GT 下一步：`form C13-C6; break C15-O17; C13-C14 order 2->1; C14-C15 order 1->2; C5-C6 order 2->1; C5-O17 order 1->2` → `CC=CC(C)C(C)(O)C(=O)C(=O)OC`
- 模型答案：`form C13-C5; form H900016-O17; break C15-O17; break H900016-O8; C13-C14 order 2->1; C14-C15 order 1->2; C5-C6 order 2->1; C6-O8 order 1->2` → `CC=CC(C)C(O)(C(C)=O)C(=O)OC`
- GT 后续：`form C13-C5; form H900016-O17; break C13-C6; break H900016-O8; C5-O17 order 2->1; C6-O8 order 1->2`
- 初判理由：合并两步
- 化学家意见：

## C098 第 1/3 步 — 初判 **acceptable**（granularity；自动类别 partial；来源 select_own, one_sample）
- 条件：reagent: NH2OH·HCl; reagent: NaOAc; solvent: MeOH; temperature: rt; reagent: sealed tube; solvent: toluene; temperature: 160 °C
- 当前物种：`N#CC=CCCCC(=O)CCCC=CC#N, NO`
- GT 下一步：`form C8-N17; form H900017-O16; form H900018-O16; break C8-O16; break H900017-N17; break H900018-N17` → `N#CC=CCCCC(CCCC=CC#N)=NO, O`
- 模型答案：`form C8-N17; C8-O16 order 2->1; charge N17 +0->+1; charge O16 +0->-1` → `N#CC=CCCCC([O-])(CCCC=CC#N)[NH2+]O`
- GT 后续：`form C12-N17; form C13-H900019; break H900019-O18; C12-C13 order 2->1; charge N17 +0->+1; charge O18 +0->-1` ‖ `form C3-C8; form C4-O18; C3-C4 order 2->1; C8-N17 order 2->1; charge N17 +1->+0; charge O18 -1->+0`
- 初判理由：肟形成经两性离子四面体中间体，GT 协同
- 化学家意见：

## C106 第 3/6 步 — 初判 **acceptable**（granularity；自动类别 partial；来源 select_own）
- 条件：solvent: toluene; temperature: 110 °C
- 当前物种：`C=Cc1ccccc1, N#[N+]C=C(C[O-])CCC=CCOCC=Cc1ccccc1`
- GT 下一步：`form H900003-O22; break C4-H900003; break C4-N3; charge N3 +1->+0; charge O22 -1->+0` → `[C]=C(CO)CCC=CCOCC=Cc1ccccc1, N#N`
- 模型答案：`form H900003-O22; break C4-H900003; C4-N3 order 1->2; N2-N3 order 3->2; charge N2 +0->-1; charge O22 -1->+0` → `[N-]=[N+]=C=C(CO)CCC=CCOCC=Cc1ccccc1`
- GT 后续：`form C4-C8; form C4-C9; C8-C9 order 2->1` ‖ `break C8-C9`
- 初判理由：先去质子成重氮烯、再失 N2，GT 协同
- 化学家意见：

## C106 第 5/6 步 — 初判 **acceptable**（granularity；自动类别 merged；来源 select_own, one_sample）
- 条件：solvent: toluene; temperature: 110 °C
- 当前物种：`C=Cc1ccccc1, OCC1=C2C(CC1)C2COCC=Cc1ccccc1, N#N`
- GT 下一步：`break C8-C9` → `OCC1=C([CH]COCC=Cc2ccccc2)[CH]CC1`
- 模型答案：`form C13-C9; form C14-C8; break C8-C9; C13-C14 order 2->1` → `OCC1=C2C3COCC3C(c3ccccc3)C2CC1`
- GT 后续：`form C13-C9; form C14-C8; C13-C14 order 2->1`
- 初判理由：双自由基开环与环加成合并
- 化学家意见：
