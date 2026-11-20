// Run with the bundled artifact-tool runtime from a workspace that resolves its node_modules.
import fs from "node:fs/promises";
import path from "node:path";
import {Workbook, SpreadsheetFile} from "@oai/artifact-tool";

const [dir,target]=process.argv.slice(2);
const read=async (...p)=>JSON.parse(await fs.readFile(path.join(dir,...p),"utf8"));
const [report,seeds,plan]=await Promise.all([read("evidence","summary.json"),read("evidence","per-seed.json"),read("plan.json")]);
const arms=plan.arms, budgets=plan.budgets;
const names={joint:"完整 JSTP",act:"ACT · ALOHA",diffusion:"Diffusion Policy · Push-T",vjepa21:"V-JEPA 2.1 · 视频表征",smol:"SmolVLA",dino:"DINO-WM · 四轨适配"};
const types={joint:"本轮原版从头重训",act:"本轮训练末端投影和任务读出",diffusion:"本轮训练末端投影和任务读出",vjepa21:"本轮训练视频表征读出",smol:"同数据旧权重，本轮重新推理和执行",dino:"同数据旧权重，本轮重新推理和执行"};
const method={
  joint:["从头训练，共同状态推演局部后果；隐式后果回到动作路径，与直接动作路径共存。","动作示范 + 四个实测光学读数。","原版训练器及损失不变；本轮没有单独验收分割。"],
  act:["官方 ALOHA 预训练；前后画面编码成一张三通道观测。原生编码器、解码器生成 100 个动作位置；取第一个，学习末端投影及四轨读出。","动作示范 + 同样四个读数。","骨干冻结；不是全参重训 ACT 或原机械臂考试。"],
  diffusion:["官方 Push-T 预训练，两帧输入。原生 U-Net、16 位动作序列、4 步确定性 DDIM；取首个可执行位置，训练末端投影及四轨读出。","动作示范 + 同样四个读数。","骨干冻结，读取最后去噪状态；不等于原场景的 100 步 DDPM。"],
  vjepa21:["官方蒸馏 ViT-B/16 编码试探前后，保留时空位置。训练读出预测四轨动作偏好与部分光学变化。","动作示范 + 同样四个读数。","冻结视频编码器；不是 V-JEPA-AC 原生规划器。"],
  smol:["官方视觉语言骨干与流动作专家；已在完全相同的三档示范量和抽样表下微调。","同一批动作示范；未加光学损失。","加载九份旧训练权重，重新推理、操作模拟器；不是本轮重新微调。"],
  dino:["预训练 DINOv2-S/14 加原作者 ViTPredictor，训练预测四轨光学变化的适配模型。","动作示范 + 同样四个读数。","加载九份旧训练权重，重新推理、操作模拟器；不是原作者原任务的规划器。"]
};
const wb=Workbook.create(), main=wb.worksheets.add("成对成绩"), detail=wb.worksheets.add("逐种子"), guide=wb.worksheets.add("比较口径");
const navy="#19374A",green="#086D63",text="#183548",muted="#4A6574",pale="#E1F2EA";
const font=(size=10,bold=false,color=text)=>({name:"Microsoft YaHei",size,bold,color});
const letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ";
function grid(s,widths,height){
  s.showGridLines=false;const z=letters[widths.length-1];const r=s.getRange("A1:"+z+height);
  r.format.font=font();r.format.rowHeightPx=26;r.format.verticalAlignment="center";
  widths.forEach((w,i)=>s.getRange(letters[i]+"1:"+letters[i]+height).format.columnWidthPx=w);
}
function value(s,a,v){s.getRange(a).values=[[v]];}
function eq(s,a,v){s.getRange(a).formulas=[[v]];}
function banner(s,row,end,txt,kind){
  const r=s.getRange("A"+row+":"+end+row);r.merge();r.values=[[txt]];
  r.format.font=kind==="title"?font(17,true,"#FFFFFF"):font(10,kind==="band",kind==="band"?green:muted);
  r.format.fill=kind==="title"?navy:kind==="band"?"#EDF5F5":"#F5F8FA";
  r.format.rowHeightPx=kind==="title"?46:30;r.format.wrapText=true;
}
function hdr(s,row,h){const r=s.getRangeByIndexes(row-1,0,1,h.length);r.values=[h];
  r.format.fill=navy;r.format.font=font(10,true,"#FFFFFF");r.format.rowHeightPx=39;r.format.wrapText=true;}
const pct="0.00%", pp="+0.00;-0.00;0.00";
const blocks=new Map();let rr=6;
grid(detail,[224,104,94,122,122,122,120,122,122,125],6+arms.length*budgets.length*3);
banner(detail,1,"J","逐种子 · 每一行是一份冻结权重","title");
banner(detail,2,"J","同一批 192 个布局在三个种子下重复执行；不要将 36,864 对当作独立布局数。","note");
hdr(detail,5,["模型","示范数","种子","成对成功数","总对数","成对成功率","单题成功数","总题数","单题成功率","相反目标同手率"]);
for(const arm of arms)for(const budget of budgets){
  const first=rr, rows=seeds.filter(v=>v.arm===arm&&v.budget===budget).sort((a,b)=>a.seed-b.seed);
  if(rows.length!==3)throw Error("Missing training seed");
  for(const d of rows){detail.getRange("A"+rr+":J"+rr).values=[[names[arm],budget,d.seed,d.won,d.pairs,null,d.single_won,d.singles,null,d.same]];
    eq(detail,"F"+rr,"=D"+rr+"/E"+rr);eq(detail,"I"+rr,"=G"+rr+"/H"+rr); rr++;}
  blocks.set(arm+":"+budget,[first,rr-1]);
  if(arm==="joint")detail.getRange("A"+first+":J"+(rr-1)).format.fill=pale;
}
for(const c of ["F","I","J"])detail.getRange(c+"6:"+c+(rr-1)).setNumberFormat(pct);
detail.freezePanes.freezeRows(5);
grid(main,[235,118,128,135,145,270],22+arms.length*2);
banner(main,1,"F","四轨实操  /  完整 JSTP 与五种外部方法","title");
banner(main,2,"F","同一段试探，两张相反的新目标卡；两次实际拉动都成功，一对才得分。","note");
banner(main,3,"F","每档都是 16,384 条物理经历；只有动作标注数改变。192 个留出布局 × 3 个训练种子。","note");
hdr(main,5,["模型","128 条示范","1,024 条示范","16,384 条示范","128 条单题成功率","这次怎样取得成绩"]);
arms.forEach((a,i)=>{
  const row=6+i;value(main,"A"+row,names[a]);value(main,"F"+row,types[a]);
  budgets.forEach((b,j)=>{const [s,e]=blocks.get(a+":"+b);eq(main,letters[j+1]+row,`=SUM('逐种子'!D${s}:D${e})/SUM('逐种子'!E${s}:E${e})`);});
  const [s,e]=blocks.get(a+":128");eq(main,"E"+row,`=SUM('逐种子'!G${s}:G${e})/SUM('逐种子'!H${s}:H${e})`);
  if(!i)main.getRange("A"+row+":F"+row).format.fill=pale;
});main.getRange("B6:E"+(5+arms.length)).setNumberFormat(pct);
const band=7+arms.length;
banner(main,band,"F","128 条示范  /  JSTP 减去对照模型（百分点）","band");
hdr(main,band+1,["对照模型","差值（点）","校正下界（点）","校正上界（点）","领先种子数（共3）","区间怎样计算"]);
arms.filter(a=>a!=="joint").forEach((a,i)=>{
  const row=band+2+i, index=6+arms.indexOf(a), c=report.contrasts.find(x=>x.budget===128&&x.other===a);
  value(main,"A"+row,names[a]);eq(main,"B"+row,`=100*($B$6-B${index})`);
  value(main,"C"+row,100*c.primary.lo);value(main,"D"+row,100*c.primary.hi);
  value(main,"E"+row,c.seed_deltas.filter(x=>x>0).length);value(main,"F"+row,"整布局配对自助法；五项比较校正");
});main.getRange("B"+(band+2)+":D"+(band+arms.length)).setNumberFormat(pp);
banner(main,band+arms.length+2,"F","这张表比较四轨任务适配，不是模型通用能力榜。外部预训练、更新参数和计算量不相同。","note");
banner(main,band+arms.length+3,"F","此处是原论文留出布局，不是重新抽取的盲考；试探由考场指定，不测自主试探。","note");
main.freezePanes.freezeRows(5);main.tabColor=green;
grid(guide,[229,385,224,330],23);
banner(guide,1,"D","实验口径  /  模型实际做了什么","title");
hdr(guide,4,["模型","输入和行动","使用的下游监督","解释边界"]);
for(const [i,a] of arms.entries()){const row=5+i;guide.getRange("A"+row+":D"+row).values=[[names[a],...method[a]]];}
guide.getRange("A5:D"+(4+arms.length)).format.wrapText=true;guide.getRange("A5:D"+(4+arms.length)).format.rowHeightPx=72;
const sourcesRow=6+arms.length;
banner(guide,sourcesRow,"D","来源与证据","band");
for(const [i,[l,u]] of [
  ["论文与原考场","https://doi.org/10.5281/zenodo.24366742"],
  ...arms.filter(a=>a!=="joint").map(a=>[names[a]+" 来源",plan[a].source]),
  ["本次统计与逐种子结果","studies/fourtrack_foundations/evidence/{summary,per-seed}.json"],
  ["先封卷再执行","studies/fourtrack_foundations/evidence/{before,after}.json"],
  ["预先锁定的本地方案","studies/fourtrack_foundations/plan.json"]
].entries()){const row=sourcesRow+1+i;value(guide,"A"+row,l);guide.getRange("B"+row+":D"+row).merge();value(guide,"B"+row,u);}
await fs.mkdir(path.dirname(target),{recursive:true});
const workbook=await SpreadsheetFile.exportXlsx(wb);await workbook.save(target);
console.log(target);

