# 发布与分发教程 · kv-cache-lab

这份教程按"**推上去 → 设置好 → 发出去 → 让它自己长**"四步走，每一步都给能直接粘的命令，
并说明**为什么这么做**。所有命令在 Windows cmd / PowerShell / Git Bash 下都成立。

---

## 阶段 0 · 发布前 5 分钟自检

> 下面用 `py` 而不是 `python`——Windows 上 `python` 常常命中 Microsoft Store 的
> 空壳存根（静默退出、EXIT=9009），这是本仓库作者踩过的坑。

**先自检，再推送。** 一个公开仓库被"改了又改"的痕迹是永久的，一次干净的首次提交更专业。

```bash
cd /d D:\WorkBuddy_Outputs\kv-cache-lab

# 1) 算术自检：不用显卡，41 项全过才说明 README 的每个数字都能从 data/ 复现
py tests/test_fit.py

# 2) 语法自检
py -m py_compile scripts/ollama_probe.py
```

期望输出：

```
all 41 checks passed - every number in README.md is reproducible from data/.
```

```bash
# 3) 检查有没有把本地路径、用户名、令牌漏进去
grep -rn "Administrator\|C:\\\\Users\|ghp_\|D:/WorkBuddy" . --include=*.py --include=*.md --include=*.yml --include=*.cff
```

期望：**没有任何输出。** 有输出就先把那一行改掉再继续。

```bash
# 4) 看一眼到底会提交哪些文件（.git 和 __pycache__ 不该出现）
git status --short
```

最终应该正好是这 15 项：

```
.gitattributes
.github/ISSUE_TEMPLATE/config.yml
.github/ISSUE_TEMPLATE/measurement.yml
.gitignore
CITATION.cff
LICENSE
README.md
README_zh.md
data/ollama_probe_f16.csv
data/ollama_probe_q8_0.csv
docs/PUBLISHING-zh.md
docs/kv-cache-report-zh.pdf
requirements.txt
scripts/ollama_probe.py
tests/test_fit.py
```

---

## 阶段 1 · 推上去

有两条路，**选一条走完，不要混着来**。

| | 路线 A：GitHub Desktop（推荐） | 路线 B：命令行 |
|---|---|---|
| 适合 | 不想碰命令 | 想精确控制、以后还要用 |
| 前置 | GitHub Desktop 3.6.5（本机已装，且已登录） | gh 2.103.3（已装在 `D:\Tools\GitHubCLI\bin`） |
| 耗时 | 约 5 分钟 | 约 3 分钟 |

> 本机已经有 GitHub Desktop 3.6.5，凭据管理器里 `git:https://github.com` 的登录也还在，
> 所以**路线 A 现在就能走，不需要再登录任何东西**。

### 1A · 图形界面路线（GitHub Desktop）

**第 1 步 · 把文件夹加进去**

打开 GitHub Desktop → 左上角 `File` → `Add local repository...` → 选
`D:\WorkBuddy_Outputs\kv-cache-lab` → 点 `Add repository`。

> 这个文件夹已经是 git 仓库了（之前 `git init` 过），所以这里不会让你"创建仓库"，
> 而是直接加进去。如果它提示"create a repository"，说明选错了文件夹，回去重选。

**第 2 步 · 提交（这一步不能跳）**

左下角会列出 **16 个文件**。在 `Summary (required)` 里填：

```
v0.1.0: measured KV cache cost per token, with confidence intervals
```

点下面的蓝色按钮 **`Commit to main`**。

**第 3 步 · 发布到 GitHub**

提交完，顶部工具栏会冒出一个按钮 **`Publish repository`**，点它。

弹窗里这样填：

| 字段 | 填什么 |
|---|---|
| Name | `kv-cache-lab` |
| Description | `KV cache quantization on an 8GB consumer GPU: 41% off the cache, 20% off the bill.` |
| **Keep this code private** | **一定要取消勾选**（勾上就变私有仓库，别人看不到） |

再点 **`Publish repository`**。等进度条走完，浏览器里就能看到你的仓库了。

**第 4 步 · 剩下两件小事**

- `Repository` 菜单 → `View on GitHub` 打开网页。
- 传封面图：网页上 `Settings` → `General` → **Social preview** → `Edit` → 上传
  `docs/social-preview.png`（仓库里已经备好，1280×640）。

### 1B · 命令行路线

（和路线 A 二选一，走完 1A 就不用看这段了。）

### 1B.1 完成 gh 授权（只需一次）

```cmd
gh auth login
```

选 `GitHub.com` → `HTTPS` → `Login with a web browser`，记下 8 位设备码贴到浏览器。
之所以要重来一遍：机器上那个旧令牌只有 `repo` 权限，gh 额外要求 `read:org`，
所以 `gh auth login --with-token` 会被拒（报 `missing required scope 'read:org'`）。

### 1B.2 提交并建仓库

```cmd
cd /d D:\WorkBuddy_Outputs\kv-cache-lab

git commit -m "v0.1.0: measured KV cache cost per token, with confidence intervals"

gh repo create kv-cache-lab --public --source=. --push
```

如果不想用 gh，就手动：github.com/new 建同名空仓库（**别勾 Add a README**），然后

```cmd
git remote add origin https://github.com/<你的用户名>/kv-cache-lab.git
git push -u origin main
```

> 推送时凭据管理器会弹窗。密码位置贴 **Personal Access Token**（`ghp_` 开头），不是账号密码。

---

## 阶段 2 · 把仓库设置好（5 分钟，决定一半的第一印象）

### 2.1 About 区：一句话 + 6 个 Topics

```cmd
gh repo edit --description "KV cache quantization on an 8GB consumer GPU: 41% off the cache, 20% off the bill. Measured, with the command that reproduces it."

gh repo edit --add-topic kv-cache,quantization,vram,ollama,llm-inference,consumer-gpu
```

**为什么**：Topics 是 GitHub 站内搜索的唯一入口；`description` 会出现在搜索结果和分享卡片上。
这两项是唯一能让陌生人"搜到你"的开关。

### 2.2 Social Preview 图（这个只能手点）

`Settings → General → Social preview → Edit → Upload`，传一张 **1280×640** 的图。

**为什么**：不传的话，仓库被分享到微信/Twitter/Slack 时是一张灰色占位卡，点击率直接腰斩。

**图已经备好了**：仓库里 `docs/social-preview.png` 就是 1280×640 的版本（深色底、双曲线、
三个数字），直接上传即可，不用自己做。

### 2.3 打 v0.1.0 Release，把 PDF 挂上去

```cmd
gh release create v0.1.0 docs/kv-cache-report-zh.pdf ^
  --title "v0.1.0 — first measurement" ^
  --notes "Two configurations (f16, q8_0) x five context lengths. A straight line gives 32.11 +/- 1.24 and 19.08 +/- 1.05 KiB/token, then fails a residual check: a one-time 67.1 MiB step at 8192 fits the same data to +/-1 MiB (F=2748) and moves the slopes to 29.99 / 17.31. Per-token overhead over theory is ~2.0 KiB, not 4.1. 15-page Chinese write-up attached."
```

**为什么**：有人只下载 PDF，不会 clone 仓库。Release 也是**后续每一次"再分发"的载体**——
加一张新显卡、补测 q4_0，都是一次 v0.2、v0.3，每次都能重新发一遍帖子。

### 2.4 确认 Cite 按钮可用

仓库首页右侧出现 **"Cite this repository"** 就说明 `CITATION.cff` 被正确解析了。
没出现 = YAML 格式错，去 `Settings → Integrations` 看报错。

### 2.5 打开 Discussions（可选但推荐）

`Settings → General → Features → Discussions` 勾上。Issues 用来报数字，Discussions 用来聊天，
两者分开，仓库不会被闲聊淹没。

---

## 阶段 3 · 发出去

### 3.1 渠道排序：先英文，后中文

**这是这份教程里最重要的一条判断。** KV cache 量化是英文 LLM infra 社区的核心议题；
中文社区更常问"我这机器能不能跑起来"，英文社区问"到底省多少、为什么和理论不一样"。
你的数据正好回答后者。

| 顺序 | 渠道 | 定位 | 建议时间（北京时间） |
|---|---|---|---|
| 1 | **Reddit r/LocalLLaMA** | 最精准的池子 | 20:00—22:00（美东早上） |
| 2 | **Hacker News**（Show HN） | "厂商说省一半，实测省两成"正对口味 | 20:00—22:00 |
| 3 | V2EX `/go/create` | 中文技术人群 | 次日 10:00 |
| 4 | 即刻 | 故事线友好，氛围不刻薄 | 次日 10:00 |
| 5 | 知乎 | 长尾 SEO | 一周后单独发长文 |
| 6 | B 站 | 成本最高 | 有余力再说 |

**为什么要卡点**：英文社区的流量按美东时间走。你在北京时间中午发，等于发在他们凌晨。

### 3.2 英文帖模板（Reddit / HN）

> **Title:** Fitting one line through a KV cache benchmark gave me 32 KiB/token. The residuals say
> that is wrong.
>
> Everyone quotes "50% off" for q8_0 KV cache. I wanted the number for my own card, so I wrote a
> ~400-line stdlib-only probe that loads a model at five context lengths and fits VRAM delta vs
> `num_ctx`.
>
> RTX 4060 Laptop (8188 MiB), Qwen2.5-1.5B, Ollama 0.40.6. The straightforward fit gives:
>
> - f16: **32.11 ± 1.24 KiB/token** (theory 28.00)
> - q8_0: **19.08 ± 1.05 KiB/token** (theory 14.88)
>
> R² = 0.9956. Looks great. It is wrong.
>
> The residuals are −17, −24, +36, +19, −15. They do not scatter, they **arc**, and the peak sits
> exactly on 8192. Add a one-time level step there and SSE drops from 2731 to **2.0**
> (F(1,2) = 2748 against a 5% critical value of 18.513). The slopes move to **29.99** and **17.31**.
> The step model predicts the 32K measurement to within **0.1 MiB**; the straight line was off by
> 14.6, and that error *is* the step it refused to model.
>
> Why it matters: a 67 MiB one-time allocation, divided by the largest context, becomes
> **2.10 KiB/token** — exactly how much the naive slope is inflated (32.11 − 29.99 = 2.12). So half
> of the "fixed ~4.1 KiB/token overhead over theory" that benchmarks like this one love to quote is
> not per-token cost at all. It is an allocation step, annualised by the regression. The real
> per-token overhead is **~2.0 KiB**.
>
> Net: q8_0 saves **42.3%** of the KV cache and **19.6% of total VRAM at 32K**. The second number is
> the one you quote to a customer.
>
> What I did not do, and say so in the README: one load per point (no repeats, so the error bars are
> optimistic), no output-quality test, and **I do not know what causes the step** — three candidate
> mechanisms are listed, none confirmed.
>
> Repo: <link> — raw CSVs plus `tests/test_fit.py`, 72 assertions that re-derive every number in the
> README without a GPU, including the F test.

**注意最后一段**：把"我的结论脆弱在哪"主动写在帖子里。这是唯一能让 HN 那群人不喷你的方式，
也是唯一能让人相信其余部分的理由。

### 3.3 中文帖模板（V2EX / 即刻）

> **标题：** 用一张 8G 笔记本显卡量 KV cache，结果发现这类测试普遍在算错一笔账
>
> 各大厂都在卷 KV cache，宣传口径是"省 50%"。我不服，写了个纯标准库的小脚本：
> 五个上下文长度、逐点加载、最小二乘拟合，把显存账算到底。
>
> 一开始特别顺利，R² = 0.9956：
> - f16：**32.11 ± 1.24 KiB/token**（理论 28.00）
> - q8_0：**19.08 ± 1.05 KiB/token**（理论 14.88）
>
> 然后我去看了残差：−17、−24、+36、+19、−15。这不是散的，是一条**弧**，峰顶正落在 8192。
> 在那个位置加一个一次性台阶，SSE 从 2731 掉到 **2.0**（F(1,2) = 2748，5% 临界值 18.513），
> 斜率变成 **29.99** 和 **17.31**。台阶模型能把 32K 的实测值预测到 **0.1 MiB** 以内，
> 而单直线差了 14.6——那个误差正好就是它拒绝建模的台阶。
>
> 为什么这事重要：8192 处那 67 MiB 的一次性分配，除以最大上下文就是 **2.10 KiB/token**，
> 而这正好是单直线被抬高的量（32.11 − 29.99 = 2.12）。所以市面上常引用的
> "固定约 4.1 KiB/token 过路费"，**一半根本不是每 token 成本**，是回归把台阶摊成了年化。
> 真正的每 token 开销约 **2.0 KiB**。
>
> 修正后：q8_0 对 KV cache 本身省 **42.3%**，对 32K 总显存省 **19.6%**——要报价，用后一个。
>
> 我没搞定的也写在 README 里：每个点只加载一次（没有纯重复，误差棒偏乐观）、
> q8_0 输出质量没测、**台阶机制不明**（列了三个候选，一个都没确认）。
>
> 脚本约 400 行、纯标准库、不要 PyTorch。仓库还有 `tests/test_fit.py`，72 项断言，
> 不插显卡就能把 README 里每个数字复算一遍，包括那个 F 检验。地址：<link>

### 3.4 发帖纪律

- **一次只发一个地方，等 2 小时看反应**。同时铺五个渠道，等于每个都发得敷衍，还没法归因。
- 回帖时**先认错再解释**。有人说"你那个 4096 台阶是 XX 原因"，先回"对，我确实没确证，你这个说法我去查"。
- 别删负面评论。技术社区里，一个被认真回复的质疑比十个赞值钱。

---

## 阶段 4 · 让它自己长（三个机制）

### 机制一：把 issue 做成数据入口

仓库里已经有 `.github/ISSUE_TEMPLATE/measurement.yml`。有人开 issue 报他自己的斜率，
你就把那条数据加进 README 的社区表。**这一步把"我的测量"变成"众包数据集"**，
仓库才有持续被 star 的理由，而不是一次性新闻。

### 机制二：每次新增数据 = 一次 Release = 一次再分发

v0.2 补 q4_0 → 发个 Release → 回原帖补一条"更新：q4_0 实测 12.0，模型预测对了"。
同一个话题可以合理地出现三次，而不是刷屏。

### 机制三：把"未解之谜"留在明面上

`4096 台阶机制未确证` 这句话**不要删**。它是钩子——会有人为了这个问题点进来、留言、甚至提 PR。
一个"什么都解释清楚了"的仓库，反而没人有话可说。

---

## 阶段 5 · 第二周

| 优先级 | 动作 | 成本 |
|---|---|---|
| 高 | 补测 6144 / 24576 两个点，把 f16 的 CI 从 ±1.24 收窄 | 1 小时（脚本不用改） |
| 高 | 补测 q4_0，验证 12.0 这个预测 | 1 小时 |
| 中 | 回应所有 issue，把社区数据加进 README 表格 | 持续 |
| 低 | q8_0 vs f16 输出质量对照（最贵，要人工判读） | 半天+ |
| 别做 | 用 HF/PyTorch 重跑一遍做交叉验证 | ⚠️ 陷阱，见下 |

---

## 常见坑（全是这次真实踩过的）

| 坑 | 症状 | 解法 |
|---|---|---|
| **GitHub releases 下载被 TLS 阻断** | `curl: (35) schannel: failed to receive handshake` | 用 `https://gh-proxy.com/https://github.com/...` 前缀；`--ssl-no-revoke` 无效 |
| **winget 下不动** | `InternetOpenUrl() failed. 0x80072efd` | 同上，走镜像后本地解压 |
| **gh 登录被拒** | `missing required scope 'read:org'` | 跑交互式 `gh auth login`，别用 `--with-token` |
| **CRLF 警告** | `LF will be replaced by CRLF` | 仓库根目录放 `.gitattributes`（已放，`* text=auto eol=lf`） |
| **PDF 太大** | 仓库 clone 变慢 | 1 MB 以内可接受；超 5 MB 就传 Release 附件而不是进 git |
| **`python` 没反应** | 静默退出，EXIT=9009 | Windows 上用 `py`，`python` 可能命中 Microsoft Store 的空壳存根 |
| **为了"完整"去装 CUDA torch** | 一天白干，零产出 | **这是本仓库最大的历史教训**：当初 694 行脚本因为机器是 CPU 版 torch 而全部报废。先花 20 分钟验证 `torch.cuda.is_available()`，再决定写不写 |
| **一次写几百行再跑** | 报错定位不到 | 先写 20 行验证环境和最小可运行路径，再往上堆 |

---

## 一页速查

```cmd
:: 自检
cd /d D:\WorkBuddy_Outputs\kv-cache-lab
py tests/test_fit.py

:: 授权（一次，只用命令行路线时）
gh auth login

:: 发布
git commit -m "v0.1.0: measured KV cache cost per token, with confidence intervals"
gh repo create kv-cache-lab --public --source=. --push

:: 设置
gh repo edit --description "KV cache quantization on an 8GB consumer GPU: 41% off the cache, 20% off the bill."
gh repo edit --add-topic kv-cache,quantization,vram,ollama,llm-inference,consumer-gpu
gh release create v0.1.0 docs/kv-cache-report-zh.pdf --title "v0.1.0 — first measurement" --notes "..."

:: 手点：Settings → Social preview 上传 docs/social-preview.png（已备好，1280x640）
```

---

**最后一条不是技术，是纪律：**

发布的价值不在"发出去"这个动作，在于你被迫把每个数字都算到能自证。
`tests/test_fit.py` 那 41 项检查，就是你以后敢在任何人面前说"这个数你可以自己验"的底气。
没有它，仓库只是一份 PPT；有了它，才是一份可以被陌生人审计的证据。
