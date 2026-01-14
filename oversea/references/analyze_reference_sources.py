import json
from urllib.parse import urlparse
from collections import Counter
import matplotlib.pyplot as plt

# 输入你的引用文件
INPUT_FILE = "references_sh_en_gemini.json"   # or references_xxx.json

# 读取引用
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    refs = json.load(f)

print(f"总引用数: {len(refs)}")

# ==========================
# 1. 提取域名 Domain
# ==========================
domains = []

for r in refs:
    url = r.get("url")
    if not url:
        continue

    # 处理 Google vertex 重定向 URL
    if "vertexaisearch.cloud.google" in url:
        # 如果 title 有真实域名，使用 title
        title = r.get("title", "")
        if "." in title:
            domains.append(title.lower().strip())
        else:
            # fallback：标记 unknown
            domains.append("vertex_redirect")
        continue

    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain:
        domains.append(domain)

# ==========================
# 2. 统计引用源出现次数
# ==========================
counter = Counter(domains)
sorted_counter = counter.most_common()

# 保存 JSON 排行
output_json = "reference_source_count3.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(sorted_counter, f, ensure_ascii=False, indent=2)

print(f"引用源排行已保存到 {output_json}")

# ==========================
# 3. 生成 Markdown 报告
# ==========================
output_md = "reference_source_ranking3.md"
with open(output_md, "w", encoding="utf-8") as f:
    f.write("# 引用来源统计报告\n\n")
    f.write(f"总引用数：**{len(domains)}**\n\n")
    f.write("## Top 引用源排行:\n\n")

    for domain, count in sorted_counter:
        f.write(f"- **{domain}**：{count} 次\n")

print(f"Markdown 报告已保存到 {output_md}")

# ==========================
# 4. 生成图表（Top 20）
# ==========================

# top_n = 20
# top_items = sorted_counter[:top_n]
# top_domains = [item[0] for item in top_items]
# top_counts = [item[1] for item in top_items]
#
# plt.figure(figsize=(12, 6))
# plt.barh(top_domains[::-1], top_counts[::-1])
# plt.xlabel("引用次数")
# plt.title(f"引用源 Top {top_n}")
# plt.tight_layout()
# plt.savefig("reference_source_chart.png")

#print("图表已保存为 reference_source_chart.png")
