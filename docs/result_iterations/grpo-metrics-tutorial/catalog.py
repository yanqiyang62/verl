"""Chinese metric explanations, checked against local verl and SwanLab source."""
CATALOG = {}
GROUPS = {
    "reward": ("01", "奖励与学习信号", "回答得分如何变成训练信号？"),
    "actor": ("02", "策略更新与稳定性", "模型改了多少，更新是否稳定？"),
    "length": ("03", "文本长度与生成", "模型生成了什么规模的回答？"),
    "mismatch": ("04", "训练与采样一致性", "生成回答的模型，与训练时重算的模型一致吗？"),
    "progress": ("05", "训练进度与版本", "训练到哪里，样本是否过期？"),
    "speed": ("06", "速度与计算效率", "每一步的时间花在哪里？"),
    "balance": ("07", "批次负载分配", "各数据并行分区的 token 负载是否均衡？"),
    "memory": ("08", "训练内存", "训练进程的显存与主机内存占用多少？"),
    "system": ("09", "系统监控", "设备与主机当前在忙什么？"),
}
SOURCES = {
    "data": ("verl/trainer/ppo/metric_utils.py", 426, 610),
    "grpo": ("verl/trainer/ppo/core_algos.py", 268, 341),
    "ppo": ("verl/trainer/ppo/core_algos.py", 1285, 1377),
    "loss": ("verl/workers/utils/losses.py", 60, 146),
    "grad": ("verl/workers/engine/fsdp/transformer_impl.py", 780, 830),
    "worker": ("verl/workers/engine_workers.py", 190, 235),
    "entropy": ("verl/trainer/ppo/v1/trainer_base.py", 1548, 1607),
    "debug": ("verl/utils/debug/metrics.py", 48, 122),
    "corr": ("verl/trainer/ppo/rollout_corr_helper.py", 942, 1009),
    "time": ("verl/trainer/ppo/metric_utils.py", 612, 688),
    "loop": ("verl/trainer/ppo/v1/trainer_base.py", 438, 489),
    "version": ("verl/trainer/ppo/v1/trainer_base.py", 1855, 1900),
    "balance": ("verl/utils/seqlen_balancing.py", 264, 304),
    "aggregate": ("verl/trainer/ppo/v1/utils.py", 20, 140),
    "agent": ("/shared/users/yangyq/data/recipe_v2_qwen38_native/tf_rl/agent_loop.py", 13, 43),
    "kl": ("verl/trainer/ppo/core_algos.py", 2188, 2280),
    "reward": ("/shared/users/yangyq/data/recipe_v2_qwen38_native/tf_rl/reward.py", 113, 175),
}
SDK = "/shared/users/yangyq/env/verl/lib/python3.12/site-packages/swanlab/sdk/internal/probe_python/hardware_vendor/"
SOURCES.update({"sysmem": (SDK + "memory.py", 38, 105), "syscpu": (SDK + "cpu.py", 28, 73), "sysgpu": (SDK + "accelerator/nvidia.py", 30, 160)})


def add(key, title, group, unit, what, formula, read, trap, related, source):
    assert key not in CATALOG, key
    CATALOG[key] = dict(key=key, title=title, group=group, unit=unit, what=what,
                        formula=formula, read=read, trap=trap, related=related.split(), source=source.split())


def family(prefix, title, group, unit, what, formula, read, trap, related, source, scope="本批样本"):
    suffixes = {"mean": ("均值", "取平均值，观察整体水平"), "max": ("最大值", "取最大值，观察上沿或极端样本"), "min": ("最小值", "取最小值，观察下沿或最弱样本")}
    for suffix, (label, meaning) in suffixes.items():
        add(prefix + "/" + suffix, title + " · " + label, group, unit,
            what + " 本图在" + scope + "中" + meaning + "。", formula + " → " + label,
            read, trap + " 图中的 min/max 是该 step 内的统计，不是截至该步的历史极值。", related, source)


family("critic/score", "原始评分", "reward", "分", "奖励函数给每条完整回答打分，再将回答内 token 分数相加；排除零长度回答。", "Sᵢ = Σₜ token_level_scoresᵢₜ",
       "本实验是参考答案一致性：0.8 × 字段匹配分 + 0.2 × 完全匹配。看均值和最小值是否一起提高，并抽查真实输出。",
       "0 分可来自格式错误、工具不匹配或字段不匹配。分数不是实际调用工具后的业务成功率；critic 前缀不代表启用了价值模型。",
       "critic/rewards/mean critic/advantages/max", "data reward")
family("critic/rewards", "训练奖励", "reward", "分", "送入优势计算的每条回答总奖励，排除零长度回答。它可能在原始 score 上加入 KL 惩罚。", "Rᵢ = Σₜ token_level_rewardsᵢₜ",
       "本实验 use_kl_in_reward=false，因此应与 score 对齐；KL 在 actor loss 中单独处理。看较长窗口及同题两回答的分差。",
       "reward=1 说明通过参考答案打分，不等于泛化能力已达标；两个回答都得 1，GRPO 相对优势仍可为 0。", "critic/score/mean critic/advantages/max actor/kl_loss", "data grpo reward")
family("critic/advantages", "相对优势", "reward", "无量纲", "衡量一个回答相对同题其他回答更好还是更差；正数鼓励，负数抑制，零表示无相对偏好。", "Aᵢ = (Rᵢ − 同题均值) / (同题样本标准差 + 10⁻⁶)，广播到有效响应 token",
       "本实验每题 2 个回答：分数不同时，优势约为 ±0.707；同分时为 0。结合 max/min 判断是否存在非零信号。",
       "均值按有效 token 统计，长回答权重更大，因此即使同题两个优势相反，图上 mean 也不必为零。mean≈0 不等于没有学习。", "critic/rewards/max critic/rewards/min actor/grad_norm", "data grpo", "有效响应 token")
family("critic/returns", "回报目标（本次等于优势）", "reward", "无量纲", "算法返回的 returns 张量。在当前 GRPO 实现中，与 advantages 使用同一个计算结果。", "GRPO: returns = advantages",
       "本次应与相应的 advantages 曲线重合，可作为一致性检查。换用 GAE 等算法后定义会改变。",
       "不要把本次 returns 当成原始累计奖励，也不要据此判断价值网络的拟合情况。", "critic/advantages/mean critic/rewards/mean", "data grpo", "有效响应 token")

add("actor/pg_loss", "策略梯度损失", "actor", "损失", "衡量当前策略是否在提高正优势回答概率、降低负优势回答概率，使用 PPO 裁剪目标。", "r = π当前 / πold；聚合 max(−rA, −clip(r,0.8,1.2)A)，负优势还有 dual-clip",
    "和优势、reward、梯度一起看。更新前 r≈1 且优势相互抵消时，损失可接近零，但梯度仍然非零。", "强化学习的目标和样本每步都在变化；pg_loss 不必像监督学习 loss 那样持续下降，也不能单独衡量回答质量。", "critic/advantages/max actor/grad_norm critic/rewards/mean", "ppo loss")
add("actor/loss", "总训练损失", "actor", "损失", "真正参与 actor 反向传播的目标，合并策略损失、参考模型 KL 和可选熵项。", "本次 loss ≈ pg_loss + 0.001 × kl_loss（entropy_coeff=0）",
    "拆开看 pg_loss 与 KL 项分别贡献多少；微批次归一化和跨设备聚合会影响日志的逐点对齐。", "数值小或为负并不意味着训练好；loss 的数值可抵消，梯度不会因此必然抵消。", "actor/pg_loss actor/kl_loss actor/grad_norm", "loss worker")
add("actor/grad_norm", "裁剪前梯度范数", "actor", "L2 范数", "所有参与训练参数的梯度合在一起有多大；本次主要是 LoRA 参数。记录的是裁剪前的总范数。", "||g||₂ = √Σⱼ gⱼ²；本次裁剪上限 1",
    "超过 1 表示会缩放梯度，不是裁剪失效。长期极小可结合组内 reward 差异、KL 和学习率定位。", "单个尖峰不能证明梯度爆炸；持续非有限值更值得排查。即便优势为零，参考 KL 项仍可能产生梯度。", "critic/advantages/max actor/lr actor/kl_loss", "grad")
add("actor/lr", "学习率", "actor", "步长系数", "优化器更新参数时使用的学习率。", "AdamW lr；本次配置 3×10⁻⁶，constant 调度",
    "这次平直曲线符合 constant 设置。它决定更新尺度，但实际参数变化还取决于梯度、动量和自适应归一化。", "不同模型、LoRA rank 与 batch 下学习率不能直接横向比较；学习率不是参数的实际变化量。", "actor/grad_norm actor/ppo_kl", "worker")
add("actor/entropy", "输出分布熵", "actor", "nat/token", "在有效响应位置上，模型对整个词表的预测有多分散。分布越集中，熵越低。", "H = −Σ词表 p(v) ln p(v)，再按响应 token 聚合",
    "持续降低表示更确定；结合 reward 和真实回答多样性判断是在学会任务还是过早变得单一。当前 v1 会在重算 old_log_prob 时统计熵。", "entropy_coeff=0 表示不把熵奖励加入损失，不代表不记录熵。熵不是回答之间的直接相似度，也不等于采样 token 的负 logprob。", "critic/rewards/mean response_length/mean rollout_corr/training_ppl", "entropy loss")
add("actor/kl_loss", "相对参考模型的 KL 惩罚", "actor", "nat/token", "衡量训练中的策略偏离固定参考策略的程度，作为正则项约束模型不要漂移太远。", "low_var_kl: d=ln πref−ln π当前；exp(d)−d−1（实现含数值截断），按 token 聚合",
    "与 reward 一起看：允许适当偏移以学习任务；持续快速增长时检查学习率、梯度与 KL 系数。", "这是当前策略与参考模型的比较，不是 actor/ppo_kl 的新旧训练策略比较，也不是 rollout_corr 的采样引擎一致性。", "actor/kl_coef actor/ppo_kl rollout_corr/k3_kl", "loss kl")
add("actor/kl_coef", "KL 惩罚系数", "actor", "系数", "参考模型 KL 项在总损失中的权重。", "总损失中的 KL 贡献 = kl_coef × kl_loss；本次 0.001",
    "固定在 0.001 符合配置；更大通常加强回到参考策略的约束。", "它不是 KL 测量值，也不是 algorithm.kl_ctrl 中用于 reward KL 的系数。当前使用 actor.kl_loss_coef。", "actor/kl_loss actor/loss", "loss")
add("actor/ppo_kl", "本次更新的新旧策略差异", "actor", "nat/token", "在该批样本上比较当前 actor 与更新前重算的 old 策略，监测一次 PPO 更新的变化。", "平均[ln πold − ln π当前]，log 比值计算时截断至 [−20,20]",
    "持续变大意味着更新幅度增大；配合裁剪比例看。本次一个 PPO epoch、小 batch，记录时新旧策略接近而长期为零也可能正常。", "有限样本近似可为负；零不证明权重完全没更新，因为记录可能发生在 optimizer.step 之前。", "actor/pg_clipfrac actor/grad_norm actor/kl_loss", "ppo")
add("actor/pg_clipfrac", "PPO 主裁剪触发比例", "actor", "比例 0–1", "有效响应 token 中，裁剪后的损失严格大于未裁剪损失的比例。它衡量主裁剪分支实际约束了多少 token。", "mean[−clip(r,0.8,1.2)A > −rA]",
    "长期偏高提示许多更新被限制，结合 ppo_kl、lr 检查更新尺度；接近零也可能只是更新很小或优势为零。", "不等于所有 r 超出 [0.8,1.2] 的比例；还依赖优势的正负。不是生成长度的 clip_ratio。", "actor/ppo_kl actor/pg_clipfrac_lower critic/advantages/max", "ppo")
add("actor/pg_clipfrac_lower", "负优势的 dual-clip 比例", "actor", "比例 0–1", "本次 vanilla PPO 中，负优势样本触发额外 dual-clip 约束的有效 token 比例。", "mean[A<0 且 max(−rA,−clip(r)A)>−3A]；clip_ratio_c=3",
    "通常很低；升高说明部分负优势 token 的概率比值异常大，需要与主裁剪、ppo_kl 一起检查。", "名字含 lower，但本实现不是单纯 r<0.8 的比例；换用其他 policy loss 时语义也可能变化。", "actor/pg_clipfrac actor/ppo_kl", "ppo")

for prefix, title, scope in [("response_length", "回答长度", "所有回答，含零长度"), ("response_length_non_aborted", "非零回答长度", "非零长度回答"), ("prompt_length", "输入长度", "所有输入")]:
    prompt = prefix == "prompt_length"
    family(prefix, title, "length", "token", "有效" + ("输入" if prompt else "响应") + " token 数，由长度字段或 attention mask 得到，不含 padding。", "L = 有效 token 数",
           ("输入长度反映任务组成与上下文负载；本次 max_prompt_length=8192。" if prompt else "长不一定好，短也不一定坏；对照 reward 和实际输出。本次 max_response_length=4096。"),
           "token 不是汉字数或字符数。" + ("输入包含模板、工具声明等上下文，不只是用户问题。" if prompt else "本次使用自定义 fixed-prefix agent，长度按其返回数据统计；不能仅凭长度判断格式正确。"),
           "critic/rewards/mean response/aborted_ratio timing_s/gen", "data", scope)
    add(prefix + "/clip_ratio", title + " · 触及张量长度边界比例", "length", "比例 0–1", "长度等于对应 prompt/response 张量最后一维宽度的样本比例。" + ("仅统计非零回答。" if "non_aborted" in prefix else ""),
        "mean[L = 张量宽度]", "结合配置上限、样本和终止原因检查是否触及长度限制。", "代码比较的是当前张量宽度，不一定总等于配置 max_length；命中边界也不证明因超长被截断，prompt 动态 padding 尤其要谨慎。", "response_length/max prompt_length/max", "data")
add("response/aborted_ratio", "零长度回答比例", "length", "比例 0–1", "当前 batch 中 response_length 为零的样本占比。", "mean[response_length = 0]",
    "上升时检查生成异常、调度中断、数据和 mask；它会让全体回答长度与非零回答长度曲线分离。", "这是按零长度定义的 aborted，不覆盖所有服务错误或所有被截断回答；不能把 0 解释为系统没有任何错误。", "response_length/mean response_length_non_aborted/mean", "data")
family("training/num_turns", "轨迹交互轮数", "length", "轮", "AgentLoop 上报每条轨迹的交互轮数，描述一次任务经过多少轮。", "统计 agent 上报的 num_turns 字段",
       "本次 fixed_prefix_agent 明确返回 num_turns=1，平直在 1 符合预期；多轮任务中结合时延和 reward 看额外轮次是否有效。", "轮次取决于 agent 的计数约定，不是 token 数或 PPO epoch 数。", "response_length/mean timing_s/gen", "data agent")

add("training/rollout_probs_diff_valid", "概率差异统计有效标记", "mismatch", "0 / 1", "是否至少存在一个有效响应 token，可用于采样与训练概率差异统计。", "1 = response_mask.any()；否则 0，相关差异返回 NaN",
    "应先看它，再解释差异和相关系数。", "1 只保证有有效 token；不保证概率一致、没有 NaN，也不保证相关系数在零方差情况下有定义。", "training/rollout_probs_diff_mean training/rollout_actor_probs_pearson_corr", "debug")
for suffix, title, formula in [("mean", "平均", "mean(|pactor − prollout|)"), ("max", "最大", "max(|pactor − prollout|)"), ("std", "标准差", "std(|pactor − prollout|)，样本标准差")]:
    add("training/rollout_probs_diff_" + suffix, "采样/训练概率差 · " + title, "mismatch", "概率差", "对同一批已采样 token，比较 rollout 记录的概率和 actor 重算的概率。", formula,
        "越接近零通常越一致。mean 看整体，max 找局部尖峰，std 看差异是否分散。", "源码注释提到 logprob，但实际先 exp 再相减，测的是概率绝对差；不是全词表的分布距离。std 在仅一个有效 token 时可能为 NaN。", "training/rollout_probs_diff_valid rollout_corr/k3_kl training/rollout_actor_probs_pearson_corr", "debug")
add("training/rollout_actor_probs_pearson_corr", "采样/训练概率相关性", "mismatch", "相关系数 −1–1", "对已生成 token 的两组概率做 Pearson 相关，衡量高低变化是否一致。", "corr(exp(old_log_probs), exp(rollout_log_probs))",
    "接近 1 表示线性变化一致；和绝对概率差一起读。", "相关系数为 1 仍可能有偏移或缩放，不能证明概率相等。零方差或样本太少时可能无定义。", "training/rollout_probs_diff_mean rollout_corr/k3_kl", "debug")
for who, label in [("training", "训练端"), ("rollout", "采样端")]:
    add("rollout_corr/" + who + "_ppl", label + "困惑度", "mismatch", "倍数", "该模型给已生成回答分配的概率有多低。每条回答先计算困惑度，再对回答求平均。", "meanᵢ exp(−mean有效token ln π" + who + ")",
        "与另一端 PPL 是否接近比绝对高低更有诊断价值。高表示模型对这些 token 更意外。", "是在自己采样的训练回答上计算，不是固定测试集困惑度；下降不自动意味着任务能力提升。", "rollout_corr/ppl_ratio rollout_corr/log_ppl_abs_diff", "corr")
    add("rollout_corr/" + who + "_log_ppl", label + "对数困惑度", "mismatch", "nat/token", "每条回答的平均负对数概率，再按回答平均；适合查看 PPL 的小变化。", "meanᵢ[−mean有效token ln π" + who + "]",
        "两端越接近，平均 log 概率越一致；无需把指数放大后的数值误当成巨大质量差异。", "平均 log(PPLᵢ) 不等于 log(平均 PPLᵢ)；也不等于全词表熵。", "rollout_corr/training_log_ppl rollout_corr/rollout_log_ppl actor/entropy", "corr")
add("rollout_corr/kl", "采样对训练的直接 KL 估计", "mismatch", "nat/token", "以 rollout 生成的 token 为样本，比较采样策略与训练端 old 策略。", "mean token[ln πrollout − ln πold]",
    "预期两端同版本时接近零；与 k3、概率差和版本滞后联合判断数值或权重不一致。", "有限样本可为负；不是固定参考模型 KL。按 token 平均，和按序列平均的 log_ppl_diff 权重不同。", "rollout_corr/k3_kl training/off_policy/trajectory_staleness/mean", "corr")
add("rollout_corr/k3_kl", "采样/训练 K3 差异估计", "mismatch", "nat/token", "使用较稳定、逐 token 非负（忽略舍入误差）的表达式衡量两端不一致。", "r = πold / πrollout；mean token[r − ln r − 1]",
    "两端完全一致时为零。与直接 KL 配合，避免只看有正负抵消的平均 log 比值。", "衡量采样端和训练端，不代表是否偏离参考模型；极端概率比会使指数项非常大。", "rollout_corr/kl training/rollout_probs_diff_max actor/kl_loss", "corr")
for suffix, label, formula in [("", "均值", "meanᵢ dᵢ"), ("_abs_diff", "绝对均值", "meanᵢ |dᵢ|"), ("_max", "最大值", "maxᵢ dᵢ"), ("_min", "最小值", "minᵢ dᵢ")]:
    key = "rollout_corr/log_ppl_abs_diff" if suffix == "_abs_diff" else "rollout_corr/log_ppl_diff" + suffix
    add(key, "对数困惑度差 · " + label, "mismatch", "nat/token", "先对每条回答计算 d = log PPL训练 − log PPL采样，再统计回答之间的差异。", formula + "；dᵢ = meanₜ(ln πrollout − ln πold)",
        "有符号差为正表示训练端对这条回答更意外，负值相反；绝对均值反映不抵消的差异。max/min 可定位两条回答是否一正一负。", "这是序列均值的统计；长度不同的回答各有一票，不等于 token 加权的 rollout_corr/kl。", "rollout_corr/kl rollout_corr/ppl_ratio", "corr")
add("rollout_corr/ppl_ratio", "训练/采样困惑度比", "mismatch", "倍数", "每条回答先求训练端 PPL 与采样端 PPL 的比值，再平均。", "meanᵢ(PPL训练ᵢ / PPL采样ᵢ) = meanᵢ exp(dᵢ)",
    "接近 1 表示困惑度相近；高于 1 意味着训练端平均更不确信这些回答。", "先求比值再平均，不等于两张平均 PPL 图相除，也不等于 exp(平均 log_ppl_diff)。", "rollout_corr/log_ppl_diff rollout_corr/training_ppl rollout_corr/rollout_ppl", "corr")
for level, title, formula in [("token", "单 token", "meanₜ[(πold / πrollout)²] − 1"), ("seq", "整条回答", "meanᵢ exp(2Σₜ ln(πold / πrollout)) − 1")]:
    add("rollout_corr/chi2_" + level, title + "的重要性权重二阶统计", "mismatch", "无量纲", "χ² 型统计，反映训练/采样概率比的二阶矩；大权重会增加重要性采样方差。", formula + "；实现对 log 比值做安全截断",
        "观察是否出现巨大尖峰；seq 会将一整条回答的概率比相乘，对长度和微小偏差特别敏感。", "有限样本估计可为负，不能强行当作精确非负散度；本次 rollout_is/rollout_rs 关闭，记录该诊断不表示实际启用了校正。", "rollout_corr/k3_kl response_length/mean", "corr")

add("training/global_step", "训练更新步", "progress", "step", "训练器记录的全局训练步编号。", "每个训练 step 记录 global_steps", "用它对齐 reward、梯度和耗时；暂停后时间继续流逝，step 未必增加。", "不等于处理的 token 数；也不是系统监控的独立采样序号。", "training/epoch timing_s/step", "loop")
add("training/epoch", "数据遍历轮次", "progress", "epoch 索引", "训练器当前数据遍历轮次的记录。", "current_epoch 随日志上报；通常从 0 开始", "本次 total_epochs=1，曲线长期为 0 并不表示没有训练；用 global_step 查看进度。", "epoch=0 不是进度为 0%；本次 shuffle=false，不同 step 的题目难度可能系统性变化。", "training/global_step critic/rewards/mean", "loop")
for kind, label, formula, read in [
    ("trajectory_spans", "轨迹跨越版本跨度", "max版本 − min版本 + 1", "同步单版本生成通常为 1。大于 1 表示一条轨迹跨越版本范围。"),
    ("trajectory_staleness", "最新生成版本滞后", "(当前训练步 − 1) − 轨迹最大版本", "同步新鲜样本通常为 0。变大说明即使轨迹最新部分也落后训练端。"),
    ("trajectory_staleness_worst", "最旧生成版本滞后", "(当前训练步 − 1) − 轨迹最小版本", "衡量轨迹最旧部分有多陈旧；结合 spans 看是否一条轨迹跨多次权重更新。")]:
    family("training/off_policy/" + kind, label, "progress", "版本步", "用轨迹携带的模型版本标签，统计采样数据与当前训练步的关系。", formula, read,
           "是版本差，不是秒数。spans 是最大与最小版本的跨度，不严格等于不同版本的去重计数；版本一致也不能排除引擎数值差异。", "rollout_corr/k3_kl training/global_step", "version")

STAGES = {
    "gen": ("生成与取样", "采样阶段取得训练轨迹的耗时，可能含队列等待及 agent 内部奖励计算。", "回答变长或生成等待都会增加它；配合 response_length 与 GPU 利用率定位。"),
    "old_log_prob": ("训练端概率重算", "更新前 actor 对已生成回答重算 old_log_probs 和熵的耗时。", "这是训练端前向，与 rollout 逐 token 生成不同。"),
    "ref": ("参考模型前向", "固定参考策略计算回答 token 的 log probability，供 KL 使用。", "对照输入/回答长度；这部分不是策略更新。"),
    "adv": ("优势计算", "根据 reward 和同题分组计算 GRPO 优势的耗时。", "通常比模型前向短；突增时检查数据传输、批量大小或调度。"),
    "update_actor": ("策略训练", "actor 前向、反向传播和优化器更新等训练阶段耗时。", "与 token 负载、显存和吞吐率一起看。"),
    "update_weights": ("采样权重同步", "把更新后的策略权重同步给 rollout 侧的耗时。", "本次同步训练会付出权重同步成本；LoRA、传输实现和设备会影响耗时。"),
    "save_checkpoint": ("保存检查点", "将训练状态写入 checkpoint 的耗时。", "本次 save_freq=20，周期性出现是预期事件；缺失 step 不是 0 秒。"),
    "step": ("整步训练", "训练器 step 计时器覆盖的一次循环耗时，包含该计时范围内的 checkpoint 保存。", "用分阶段耗时解释峰值；本地源码在 step 计时结束后还有验证与日志操作，因此不等于所有外部墙钟开销。"),
}
for key, (title, what, read) in STAGES.items():
    add("timing_s/" + key, title + "耗时", "speed", "秒", what, "对应 marked_timer 的秒数", read,
        "计时器可能嵌套或涵盖等待，不能把所有 timing_s 无条件相加；这是每步/本次聚合耗时，不是全程累计。", "perf/time_per_step timing_s/gen timing_s/update_actor", "time loop")
for key in ["gen", "ref", "adv", "update_actor"]:
    title, what, read = STAGES[key]
    denom = "本 batch 响应 token 总数" if key == "gen" else "本 batch 输入+响应 token 总数"
    add("timing_per_token_ms/" + key, title + " · 每 token 摊销耗时", "speed", "ms/token", "把阶段总时间除以处理的 token 数，用于减少批次长度变化的影响。", "1000 × timing_s/" + key + " ÷ " + denom,
        "与同阶段原始时间配合看：总时间变长但每 token 时间稳定，可能主要是工作量增加。", "gen 的分母只有响应，其余阶段用输入+响应；不应跨阶段直接比较效率。它是批量摊销值，不是单个用户的流式 token 延迟。", "timing_s/" + key + " response_length/mean perf/total_num_tokens", "time")
add("perf/total_num_tokens", "本步有效 token 总量", "speed", "token", "该批所有序列的有效 token 总数，来自 global_token_num。", "Σ global_token_num（输入+响应）", "用于解释时间与吞吐率的变化；和单条回答长度不同。", "是本批工作量，不是自训练开始累计的 token 数，也不是只生成的 token。", "perf/throughput perf/time_per_step", "time")
add("perf/time_per_step", "每步耗时", "speed", "秒", "吞吐率计算所用的整步时间。", "perf/time_per_step = timing_s/step", "与 step 曲线应一致；保存 checkpoint 的步往往更慢。", "训练越来越快可能来自回答更短，要结合长度与 reward 判断。", "timing_s/step perf/throughput", "time")
add("perf/throughput", "每 GPU 吞吐率", "speed", "token/s/GPU", "整步平均每 GPU 每秒处理的有效 token 数。", "total_num_tokens ÷ (time_per_step × GPU数量)", "同模型、长度和硬件条件下越高通常越高效；本次配置 1 GPU。", "包含输入 token，不是 vLLM 单独的生成吞吐率；GPU 计数与 MIG/整卡硬件语义也要区分。", "perf/total_num_tokens timing_s/gen timing_s/update_actor", "time")
add("perf/mfu/actor", "Actor 模型计算利用率估计", "speed", "比例（估计）", "估算 actor 计算的 FLOPs 与设备理论峰值能力的比值。", "estimated_flops / promised_flops / world_size", "观察同配置下相对变化，结合 update_actor 时间和 token 量。", "不是 NVML GPU busy%；LoRA、MIG、模型 FLOPs 和硬件峰值估算会影响准确度，不能把它当精确硬件效率。", "timing_s/update_actor __swanlab__.gpu.0.pct", "worker")

for key, label, formula in [("min", "分配前最小负载", "min 原始分区 token 总数"), ("max", "分配前最大负载", "max 原始分区 token 总数"), ("mean", "分区平均负载", "总 token 数 / 分区数"), ("minmax_diff", "分配前负载差", "max − min"), ("balanced_min", "平衡后最小负载", "min 重新分区后 token 总数"), ("balanced_max", "平衡后最大负载", "max 重新分区后 token 总数")]:
    add("global_seqlen/" + key, label, "balance", "token/分区" if key != "minmax_diff" else "token", "按数据并行分区统计整批序列长度之和，描述负载分配。", formula,
        "多分区下，balanced_max 与 balanced_min 接近意味着分配更均衡。本次 1 GPU 且 balance_batch=false，优先看时间和长度。", "不是单条序列的 min/max；指标出现在项目中或本次有历史值，不证明当前配置启用了负载平衡。单分区相等是自然结果。", "perf/total_num_tokens timing_s/update_actor", "balance")
for key, title, what, formula, trap in [
    ("max_memory_allocated_gb", "训练张量显存峰值", "PyTorch 分配给张量的显存峰值。", "torch.cuda.max_memory_allocated() / 1024³", "这是分配器峰值，窗口取决于何时 reset；不是整个设备当前全部显存。"),
    ("max_memory_reserved_gb", "训练显存保留峰值", "PyTorch 缓存分配器向设备保留的显存峰值，包含暂未使用的缓存。", "torch.cuda.max_memory_reserved() / 1024³", "reserved 高于 allocated 不自动意味着泄漏；包含可复用缓存。"),
    ("cpu_memory_used_gb", "主机已用内存", "训练 worker 读取的系统已用 RAM。", "psutil.virtual_memory().used / 1024³", "虽然前缀是 actor/perf，但它不是 actor 进程专属 RSS，也不是 GPU 显存。")]:
    add("actor/perf/" + key, title, "memory", "GiB", what, formula, "结合序列长度、批次大小和系统监控看占用是否持续增长。峰值曲线不下降可以是峰值统计本身导致。", trap, "__swanlab__.gpu.0.mem.value __swanlab__.mem.proc", "worker")

SYSTEM = [
    ("mem.pct", "系统内存使用率", "%", "整台主机的物理内存使用比例。", "psutil.virtual_memory().percent", "sysmem", "不是当前训练进程使用率；共享主机上可能包含其他任务。"),
    ("mem.proc", "记录进程常驻内存", "MiB", "运行 SwanLab 采集器的当前进程 RSS，不含换出内存。", "Process().memory_info().rss / 1024²", "sysmem", "不是所有 Ray worker 内存之和；采集器可能在协调进程，不能代表完整训练作业。"),
    ("mem.proc.pct", "记录进程内存比例", "%", "当前采集进程 RSS 占主机总物理内存的百分比。", "Process().memory_percent()", "sysmem", "分母是主机内存；不是容器或作业的内存配额。0.03 表示 0.03%，不是 3%。"),
    ("mem.proc.avail", "主机可用内存", "MiB", "主机可供新分配使用的内存估计量。", "virtual_memory().available / 1024²", "sysmem", "名字有 proc，但实现读的是系统 available，不是当前进程专属剩余额度。"),
    ("cpu.pct", "主机 CPU 利用率", "%", "采样间隔内主机所有逻辑 CPU 的平均忙碌比例。", "psutil.cpu_percent(interval=None)", "syscpu", "不是训练进程 CPU%；多核主机平均很低时，也可能有单个核心成为瓶颈。"),
    ("cpu.thds", "记录进程线程数", "线程", "当前 SwanLab 采集进程的线程数量。", "Process().num_threads()", "syscpu", "线程数不等于正在运行的线程，也不是 GPU 线程数。"),
    ("gpu.0.pct", "GPU 0 忙碌时间比例", "%", "NVML 采样间隔内，GPU 执行 kernel 的时间比例。", "nvmlDeviceGetUtilizationRates(handle).gpu", "sysgpu", "100% 不等于 FLOPs 跑满，也不等于 MFU=1；与内存带宽、kernel 类型有关。"),
    ("gpu.0.mem.pct", "GPU 0 显存占用率", "%", "被 NVML 监控设备已用显存占其总显存的百分比。", "used / total × 100", "sysgpu", "可能是物理整卡，包含其他进程或其他 MIG 实例；不能直接除以实验名中的 67G。"),
    ("gpu.0.mem.value", "GPU 0 已用显存", "MiB", "NVML 设备级已用显存容量。", "nvmlDeviceGetMemoryInfo(handle).used >> 20", "sysgpu", "与 actor 的 PyTorch 显存峰值范围不同。这里约 20 万 MiB 不表示该 MIG 训练进程单独用了 200 GiB。"),
    ("gpu.0.mem.time", "GPU 0 显存读写忙碌比例", "%", "采样间隔内，全局显存发生读或写的时间比例。", "nvmlDeviceGetUtilizationRates(handle).memory", "sysgpu", "不是显存容量占用率，也不是内存带宽达到理论峰值的百分比；零值可受采样或设备支持影响。"),
    ("gpu.0.power", "GPU 0 功率", "W", "NVML 返回的设备功率，毫瓦转瓦。", "nvmlDeviceGetPowerUsage(handle) / 1000", "sysgpu", "是设备功率读数，不是当前训练作业单独的用电功率，也不是累计耗电量 kWh。"),
    ("gpu.0.temp", "GPU 0 温度", "°C", "NVML 返回的 GPU 温度传感器读数。", "nvmlDeviceGetTemperature(handle, GPU)", "sysgpu", "正常温度范围依型号、散热和负载而异；不在这里设置一个通用报警阈值。"),
]
for key, title, unit, what, formula, source, trap in SYSTEM:
    add("__swanlab__." + key, title, "system", unit, what, formula,
        "按时间读图，与训练阶段的时间戳对应。系统监控独立采样，即使没有新的训练 step 也可能继续记录。" + ("持续升温且吞吐下降时，再检查降频和散热。" if key.endswith("temp") else "先看趋势和同一时间的关联指标。"),
        trap, "timing_s/step actor/perf/max_memory_allocated_gb", source)
