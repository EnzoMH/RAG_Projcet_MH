import re

def extract_toc(text):
    toc = []
    lines = text.split('\n')
    for line in lines:
        if re.match(r'^[Ⅰ-Ⅶ]\. ', line):
            toc.append(('main', line.strip()))
        elif re.match(r'^\d+\. ', line):
            toc.append(('sub', line.strip()))
    return toc

def summarize_section(text):
    summaries = {}
    current_section = None
    section_content = []
    lines = text.split('\n')
    
    for line in lines:
        if re.match(r'^[Ⅰ-Ⅶ]\. ', line):
            if current_section:
                summaries[current_section] = ' '.join(section_content)
            current_section = line.strip()
            section_content = []
        elif current_section:
            section_content.append(line.strip())
    
    if current_section:
        summaries[current_section] = ' '.join(section_content)
    
    return summaries

def analyze_proposal(text):
    toc = extract_toc(text)
    summary = summarize_section(text)
    return toc, summary

# 사용 예
toc, summary = analyze_proposal(rfp_text)

print("목차:")
for level, item in toc:
    print(f"{'  ' if level == 'sub' else ''}{item}")

print("\n섹션 요약:")
for section, content in summary.items():
    print(f"{section}:")
    print(f"{content[:200]}...\n")  # 각 섹션의 처음 200자만 출력