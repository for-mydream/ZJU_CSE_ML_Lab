import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell
import os

def notebook_to_markdown(notebook_path, output_path):
    # 读取Jupyter Notebook文件
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # 创建一个新的Markdown文件
    markdown_content = ""
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            # 直接添加Markdown cell的内容
            markdown_content += cell.source + "\n"
        elif cell.cell_type == 'code':
            # 将代码cell转换为Markdown格式
            markdown_content += "```python\n"
            markdown_content += cell.source + "\n"
            markdown_content += "```\n"
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存Markdown文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Markdown file saved as {output_path}")

# 使用函数
notebook_path = 'main (4).ipynb'  # 替换为你的Notebook文件路径
output_path = '作家风格识别.md'  # 替换为你希望保存Markdown文件的路径和文件名
notebook_to_markdown(notebook_path, output_path)