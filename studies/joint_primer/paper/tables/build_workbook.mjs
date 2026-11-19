import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [root, target] = process.argv.slice(2);
const study = path.join(root, "studies", "joint_primer");
const json = async (name) => JSON.parse(await fs.readFile(path.join(study, name), "utf8"));
const [main, official, seedsM, seedsO, extra] = await Promise.all([
  "evidence/analysis/summary.json", "evidence/official/analysis/summary.json",
  "evidence/analysis/per-seed.json", "evidence/official/analysis/per-seed.json",
  "evidence/challenge/results.json"
].map(json));
const models = ["joint","world","aux","masked","dino","direct","flow","smol"];
const labels = {
  joint:"JSTP 共享转移", world:"一步世界模型", aux:"同结构，断开后果回输", masked:"掩码轨迹 Transformer",
  dino:"DINO-WM 任务适配", direct:"仅直接动作", flow:"条件流动作", smol:"官方 SmolVLA 局部微调",
  "world-isolated":"世界模型，隔离动作梯度"
};
const methods = {
  joint:["同一个转移单元先形成拉 A 的隐式预期，再预测自身动作；历史信息也可直接进入动作读出。","同时学动作和局部光学位移。正常会计算参考预期，不要求生成整帧，也不逐一比较两个候选。","检验少量动作标注下，局部后果能否帮模型使用试探经历来完成新目标。"],
  world:["分别预测竖拉 A、B 的光学后果，再用学到的目标评分挑一个。","学习光学位移。主考场允许动作误差回传给动力学预测器。","显式比较两个后果是强对照。128 条档里，有一个种子胜过对应 JSTP。"],
  aux:["与 JSTP 共享编码和转移结构，但选把手时不读取参考后果的隐状态。","同时学动作和光学位移，保留两种损失，只断开预期到动作的通路。","最接近的机制对照：能读出共动，不等于能把这份预期直接用在动作上。"],
  masked:["用不同查询从同一个单层 Transformer 补出缺失的动作或光学读数。","同时学动作和光学位移。按 MTM 思路本地实现，未用原作者权重。","能检验多任务轨迹补全本身是否已经足够。"],
  dino:["冻结官方 DINOv2-S/14 视觉编码器，用 DINO-WM 作者的 ViT 预测器估计两只把手的后果，再比较。","在装置上重训预测器和任务读出；同 JSTP 学习抽到的四个光学位置，更新 107,024 个参数。","是基于官方组件的任务适配，不是原作者在这台装置上发布的完整策略。"],
  direct:["只从试探历史和目标卡直接预测选哪只把手。","只学正确动作；数据池相同，不以光学后果为训练目标。","用于区分仅模仿正确动作与学习操作后果。不能据此排除动作本身也是预测。"],
  flow:["以历史和目标为条件，从固定起点经八步流场得到有符号动作。","使用动作标注训练条件流，不设光学预测损失。","是从零训练的条件流策略，不等于某个基础 VLA。"],
  smol:["用官方 SmolVLA 视觉语言主干和动作专家，从零噪声走四步得到有符号动作。","保留原生动作训练；微调状态投影、动作输出投影和最后一层动作专家，更新 6,067,072 个参数。","同示范、同更新次数；没有额外加光学监督。属于本任务局部微调成绩。"],
  "world-isolated":["与一步世界模型一样比较 A、B，但阻断动作损失向动力学回传。","动力学仍按相同光学位移学习；另训九份。","只在第二批 192 个布局实操，不能把它与原主考场的数值直接相减。"]
};
const budgets = [128,1024,16384], ids = [9301,9419,9547];
const cols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const wb = Workbook.create();
const summary = wb.worksheets.add("同场总表");
const guide = wb.worksheets.add("架构与补考");
const detail = wb.worksheets.add("逐种子记录");
const ink = "#142D40", brand = "#234868", accent = "#1C695D", quiet = "#486171";
const typeface = "Microsoft YaHei";
const font = (size=10, bold=false, color=ink) => ({name:typeface,size,bold,color});
function grid(sh, widths, endRow) {
  sh.showGridLines = false;
  const all = sh.getRange(`A1:${cols[widths.length-1]}${endRow}`);
  all.format.font=font(); all.format.rowHeightPx=26; all.format.verticalAlignment="center";
  all.format.wrapText=false;
  widths.forEach((w,i)=>{sh.getRange(`${cols[i]}1:${cols[i]}${endRow}`).format.columnWidthPx=w;});
  sh.getRange(`A1:${cols[widths.length-1]}1`).format.rowHeightPx=10;
}
function at(sh, a, value){ sh.getRange(a).values=[[value]]; }
function line(sh, row, value){ at(sh,"A"+row,value); sh.getRange("A"+row).format.font=font(10,false,quiet); }
function title(sh, value){ at(sh,"A2",value); const r=sh.getRange("A2"); r.format.font=font(16,true,ink);r.format.rowHeightPx=37; }
function band(sh,row,value,n){
  const r=sh.getRange(`A${row}:${cols[n-1]}${row}`); r.format.rowHeightPx=33;
  r.format.borders={bottom:{style:"thin",color:"#83A1A9"}};
  at(sh,"A"+row,value); sh.getRange("A"+row).format.font=font(12,true,accent);
}
function header(sh,row,values){
  const r=sh.getRangeByIndexes(row-1,0,1,values.length);r.values=[values];
  r.format.fill=brand; r.format.font=font(10,true,"#FFFFFF");
  r.format.horizontalAlignment="center";r.format.verticalAlignment="center";r.format.wrapText=true;r.format.rowHeightPx=43;
  r.format.borders={insideVertical:{style:"thin",color:"#FFFFFF"}};
}
function stripe(sh,from,to,n,highlightFirst=false){
  for(let i=from;i<=to;i++){
    const r=sh.getRange(`A${i}:${cols[n-1]}${i}`);
    if((i-from)%2===1)r.format.fill="#F2F5F7";
    if(i===from&&highlightFirst){r.format.fill="#DDEFE9";r.format.font=font(10,true,ink);}
  }
}
function form(sh, cell, text){sh.getRange(cell).formulas=[[text]];}
const pct="0.00%", pp='+0.00;-0.00;0.00';

// Original fixed-seed records. Only the published main and external fits share this panel.
grid(detail,[234,104,97,105,105,121,124,126,180,133],85);
title(detail,"逐种子记录：主考场 72 份冻结权重");
line(detail,3,"每行一份权重，每份权重 12,288 对相反目标；同一布局内的重复不是独立样本。");
line(detail,4,"内部分组未在两种官方适配上测量；“未测”不等于 0。来源列的原始记录见表尾。");
header(detail,5,["模型","示范数","种子","成对成功率","单题成功率","严格共动读出","隐去试探后\n共动读出","相反目标\n仍选同手","同手率来源","数据来源"]);
const key=(arm,budget,seed)=>`${arm}|${budget}|${seed}`;
const lookupM=new Map(seedsM.map(r=>[key(r.arm,r.budget,r.seed),r]));
const lookupO=new Map(seedsO.map(r=>[key(r.arm,r.budget,r.seed),r]));
const data=[];const derive=[];
for(const arm of models)for(const b of budgets)for(const seed of ids){
  const k=key(arm,b,seed), m=lookupM.get(k), o=lookupO.get(k), src=m??o;
  const row=6+data.length;
  data.push([labels[arm],b,seed,src.metrics.pair,src.metrics.single,m?m.metrics.group:"未测",
    m?m.metrics.blind_group:"未测",o?o.metrics.same_action:null,o?"封存动作直接计数":"由单题与成对成功率换算",
    m?(o?"主、追两表":"主试验"):"官方适配追加"]);
  if(!o)derive.push(row);
}
detail.getRangeByIndexes(5,0,data.length,10).values=data;
for(const i of derive)form(detail,"H"+i,`=2*(E${i}-D${i})`);
const last=5+data.length;
stripe(detail,6,last,10);
for (const a of ["D","E","F","G","H"])detail.getRange(`${a}6:${a}${last}`).setNumberFormat(pct);
detail.getRange(`B6:B${last}`).setNumberFormat("#,##0");detail.freezePanes.freezeRows(5);
const base="https://github.com/ykykykk/jstp-intervention-bench/blob/1b25a5f/studies/joint_primer/";
at(detail,"A79","主试验 · 每个种子");at(detail,"B79",base+"evidence/analysis/per-seed.json");
at(detail,"A80","官方适配 · 每个种子");at(detail,"B80",base+"evidence/official/analysis/per-seed.json");
at(detail,"A81","主试验 · 布局配对区间");at(detail,"B81",base+"evidence/analysis/summary.json");
at(detail,"A82","官方适配 · 布局配对区间");at(detail,"B82",base+"evidence/official/analysis/summary.json");
at(detail,"A83","独立补考 · 另 192 个布局");at(detail,"B83",base+"evidence/challenge/results.json");
at(detail,"A84","导论 · 方法与指标定义");at(detail,"B84",base+"paper/manuscript.md");
at(detail,"A85","官方适配 · 训练与考试协议");at(detail,"B85",base+"official-plan.json");

grid(summary,[236,110,110,110,122,117,117,126],32);summary.tabColor=brand;
title(summary,"JSTP 与七个对照：三档示范量的实操成绩");
line(summary,3,"四轨、两只把手。同一段试探结束后给两张相反的未见目标卡；两次实际操作都成功，一对才得分。");
line(summary,4,"192 个训练未见布局，三个固定训练种子。所有模型来自同一份 16,384 条物理经历，三档只改变可用动作标注的数量。");
band(summary,5,"成对成功率，以及 128 条示范时 JSTP 的领先幅度",8);
header(summary,6,["模型",128,1024,16384,"128 条领先\n（百分点）","校正区间下界\n（百分点）","校正区间上界\n（百分点）","JSTP 领先的\n种子数"]);
summary.getRange("B6:D6").setNumberFormat('#,##0" 条"');
band(summary,19,"128 条示范：三个种子的成绩，以及分组和换手行为",8);
header(summary,20,["模型",...ids,"单题成功率","相反目标\n仍选同手","严格共动读出","隐去试探后\n共动读出"]);
const srcCol=c=>`'逐种子记录'!$${c}$6:$${c}$${last}`;
const mean=(out, name, b, seed)=>`=AVERAGEIFS(${srcCol(out)},${srcCol("A")},${name},${srcCol("B")},${b}${seed?`,${srcCol("C")},${seed}`:""})`;
const contrasts=[...main.paired,...official.paired].filter(x=>x.budget===128);
models.forEach((arm,index)=>{
  const r=7+index,d=21+index;
  at(summary,"A"+r,labels[arm]); form(summary,"A"+d,`=A${r}`);
  for(const c of ["B","C","D"]){
    form(summary,c+r,mean("D",`$A${r}`,`${c}$6`));
    form(summary,c+d,mean("D",`$A${d}`,"$B$6",`${c}$20`));
  }
  if(arm!=="joint"){
    const m=contrasts.find(x=>x.other===arm).family;
    form(summary,"E"+r,`=($B$7-B${r})*100`);
    at(summary,"F"+r,100*m.lo);at(summary,"G"+r,100*m.hi);
    form(summary,"H"+r,`=IF(B$21>B${d},1,0)+IF(C$21>C${d},1,0)+IF(D$21>D${d},1,0)`);
  }else for(const c of ["E","F","G","H"])at(summary,c+r,"—");
  for(const [to,from] of [["E","E"],["F","H"],["G","F"],["H","G"]]){
    if((arm==="smol"||arm==="dino")&&(to==="G"||to==="H"))at(summary,to+d,"未测");
    else form(summary,to+d,mean(from,`$A${d}`,"$B$6"));
  }
});
stripe(summary,7,14,8,true);stripe(summary,21,28,8,true);
summary.getRange("B7:D14").setNumberFormat(pct);summary.getRange("E7:G14").setNumberFormat(pp);
summary.getRange("H7:H14").setNumberFormat('0"/3"');summary.getRange("B21:H28").setNumberFormat(pct);
for(const c of ["B","C","D"]){
  summary.getRange(`${c}7:${c}14`).conditionalFormats.addCustom(`=${c}7=MAX(${c}$7:${c}$14)`,{fill:"#C2E3D0",font:{bold:true,color:"#145A43"}});
}
line(summary,15,"差值用未舍入的原数计算。主试验五个对照各列 99% 区间；官方适配两项各列 97.5% 区间。区间按整布局配对重采样。");
line(summary,16,"基线差距取 JSTP 减对手；种子数表示在三个相同初始化编号的比较中领先几次。外部预训练、光学监督与计算量不相同。");
line(summary,17,"128 条时 JSTP 平均居首；另两档断开后果回输的版本略高。相同训练示范不等于相同训练信号。");
line(summary,29,"严格共动：用独立验证布局校准线性读出器；同一装置在两种离合下，四轨方向必须全部读对。不是物体轮廓或跨遮挡身份。");
line(summary,30,"隐去试探指遮掉试探期间的亮度变化与横向位移。相反卡的正确动作恰好相反，因此同手率也等于 2 ×（单题成功率 − 成对成功率）。");
line(summary,31,"JSTP、DINO-WM 等采用与目标符号有关的评分，易随目标反号换手；是否换对方向，要看成对成功率。");
line(summary,32,"JSTP 的分组读出在三个初始化间从 34.77% 到 100.00%。表中置信区间固定这三个种子，不包括重新抽训练样本的方差。");

grid(guide,[234,354,344,340],32);title(guide,"决策方法、监督差异与单列的补充考场");
line(guide,3,"示范是由两次实测后果和目标卡导出的正确动作，不是人类演示录像。试探哪只把手在本次横向比较中由考场指定。");
line(guide,4,"三档均为 768 次更新、每次 128 个动作标签。模型看相同的目标和试探信息，但输入编码、预训练和训练目标可以不同。");
header(guide,5,["模型","怎么选把手","怎么学","如何解释这项对比"]);
const all=[...models,"world-isolated"];
guide.getRangeByIndexes(5,0,all.length,4).values=all.map(x=>[labels[x],...methods[x]]);
stripe(guide,6,14,4,true);const methodsRange=guide.getRange("A6:D14");methodsRange.format.wrapText=true;methodsRange.format.rowHeightPx=67;
band(guide,17,"补充考场：另一批 192 个布局，不与主考场混算",4);
header(guide,18,["模型",128,1024,16384]);guide.getRange("B18:D18").setNumberFormat('#,##0" 条"');
const otherArms=["joint","world","aux","masked","direct","flow","world-isolated"];
guide.getRange("A19:D25").values=otherArms.map(a=>[labels[a],...budgets.map(b=>extra.rows.find(r=>r.arm===a&&r.budget===b).pair.value)]);
guide.getRange("B19:D25").setNumberFormat(pct);stripe(guide,19,25,4,true);
line(guide,26,"上述仅为补考成对成功率；这批布局上没有两种官方适配成绩。其余六组复用冻结权重，隔离梯度对照另行训练。");
line(guide,28,"四轨实验给定了部件地址、动作原语和目标格式。主试验只问训练未见的布局与目标组合，不是现实机器人或通用视觉的排名。");
line(guide,29,"功能性分组回答“拉它时哪些轨道会一起变化”；任务成功回答“面对现在的目标，选哪只把手”。两者不能互作替代指标。");
line(guide,30,"在主试验的 128 条档，定向修改 JSTP 的后果隐状态使 11.75% 的应换手题由原先正确转向供体决定，等幅无关扰动为 5.79%。");
line(guide,31,"这种干预会使部分原本正确的动作变错，说明预期参与了决策，不能解读成提高通关率。");
wb.recalculate();
await fs.mkdir(path.dirname(target),{recursive:true});
const x=await SpreadsheetFile.exportXlsx(wb);await x.save(target);
console.log(target);
