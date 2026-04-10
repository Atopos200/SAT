"""
将 .pt 文件内容解析并生成可交互的 HTML 可视化页面。
用法: conda activate sat_cpu && cd SAT && python visualize_pt.py
会在当前目录生成 pt_viewer.html，浏览器打开即可。
"""
import os
import json
import torch
import numpy as np
from sklearn.decomposition import PCA

CKPT_DIR = os.path.join("checkpoints_cpu", "FB15k-237N")


def inspect_pt(path):
    """加载 .pt 文件，返回结构描述"""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj, describe(obj, name=os.path.basename(path))


def describe(obj, name="root", depth=0):
    """递归描述 PyTorch 对象结构"""
    info = {"name": name, "type": type(obj).__name__}
    if isinstance(obj, torch.Tensor):
        info["shape"] = list(obj.shape)
        info["dtype"] = str(obj.dtype)
        info["min"] = float(obj.float().min())
        info["max"] = float(obj.float().max())
        info["mean"] = float(obj.float().mean())
        info["std"] = float(obj.float().std()) if obj.numel() > 1 else 0
        info["numel"] = obj.numel()
    elif isinstance(obj, dict):
        info["keys"] = list(obj.keys())[:50]
        info["children"] = [describe(v, k, depth + 1) for k, v in list(obj.items())[:20]]
    elif isinstance(obj, (list, tuple)):
        info["length"] = len(obj)
        info["children"] = [describe(v, f"[{i}]", depth + 1) for i, v in enumerate(obj[:10])]
    elif hasattr(obj, '__dict__'):
        attrs = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        info["attrs"] = list(attrs.keys())[:20]
        info["children"] = [describe(v, k, depth + 1) for k, v in list(attrs.items())[:20]]
    else:
        info["value"] = str(obj)[:200]
    return info


def embedding_to_2d(tensor, n_samples=2000):
    """用 PCA 把高维嵌入降到 2D"""
    data = tensor.detach().float().numpy()
    if len(data) > n_samples:
        idx = np.random.choice(len(data), n_samples, replace=False)
        data = data[idx]
    else:
        idx = np.arange(len(data))
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(data)
        explained = pca.explained_variance_ratio_.tolist()
    else:
        coords = data
        explained = [1.0, 0.0]
    return coords.tolist(), idx.tolist(), explained


def state_dict_layer_sizes(state_dict):
    """统计 state_dict 每层参数量"""
    layers = []
    for k, v in state_dict.items():
        layers.append({
            "name": k,
            "shape": list(v.shape),
            "params": int(v.numel()),
            "dtype": str(v.dtype),
            "mb": round(v.numel() * v.element_size() / 1024 / 1024, 3)
        })
    return layers


def build_html(file_infos, embedding_data, layer_data):
    """生成完整 HTML"""
    # 预构建 HTML 片段
    overview_cards = ""
    for fi in file_infos:
        overview_cards += (
            '<div class="card">'
            '<h3>' + fi['name'] + '</h3>'
            '<div class="stat"><span class="stat-label">文件大小</span><span class="stat-value">' + fi['size'] + '</span></div>'
            '<div class="stat"><span class="stat-label">类型</span><span class="stat-value">' + fi['top_type'] + '</span></div>'
            '<div class="stat"><span class="stat-label">内容摘要</span><span class="stat-value">' + fi['summary'] + '</span></div>'
            '</div>\n'
        )

    layer_rows = ""
    for l in layer_data:
        layer_rows += (
            '<tr class="layer-row" data-name="' + l['name'].lower() + '">'
            '<td style="font-family:monospace;font-size:12px">' + l['name'] + '</td>'
            '<td class="shape">' + str(l['shape']) + '</td>'
            '<td>' + f"{l['params']:,}" + '</td>'
            '<td>' + str(l['mb']) + '</td>'
            '<td><div class="bar-container"><div class="bar" style="width:' + str(l['pct']) + '%"></div></div></td>'
            '</tr>\n'
        )

    total_params = sum(l['params'] for l in layer_data)
    total_mb = sum(l['mb'] for l in layer_data)
    n_points = embedding_data['n_points']
    exp0 = f"{embedding_data['explained'][0]:.1%}"
    exp1 = f"{embedding_data['explained'][1]:.1%}"

    emb_json = json.dumps(embedding_data)
    struct_json = json.dumps([fi['struct'] for fi in file_infos])
    names_json = json.dumps([fi['name'] for fi in file_infos])

    return (
        '<!DOCTYPE html>\n<html lang="zh">\n<head>\n<meta charset="UTF-8">\n'
        '<title>SAT .pt File Viewer</title>\n<style>\n'
        '* { margin: 0; padding: 0; box-sizing: border-box; }\n'
        'body { font-family: -apple-system, "Helvetica Neue", sans-serif; background: #0f0f23; color: #e0e0e0; padding: 20px; }\n'
        'h1 { color: #00d4ff; margin-bottom: 20px; font-size: 24px; }\n'
        'h2 { color: #ffd700; margin: 20px 0 10px; font-size: 18px; }\n'
        'h3 { color: #7fdbca; margin: 15px 0 8px; font-size: 15px; }\n'
        '.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }\n'
        '.card { background: #1a1a2e; border-radius: 12px; padding: 20px; border: 1px solid #2a2a4a; }\n'
        '.card:hover { border-color: #00d4ff; }\n'
        '.stat { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #2a2a4a; }\n'
        '.stat-label { color: #888; } .stat-value { color: #00d4ff; font-family: monospace; }\n'
        'table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }\n'
        'th { background: #16213e; color: #ffd700; padding: 8px; text-align: left; position: sticky; top: 0; }\n'
        'td { padding: 6px 8px; border-bottom: 1px solid #1a1a2e; }\n'
        'tr:hover td { background: #16213e; }\n'
        '.bar-container { width: 100%; background: #2a2a4a; border-radius: 4px; height: 14px; }\n'
        '.bar { height: 14px; background: linear-gradient(90deg, #00d4ff, #7fdbca); border-radius: 4px; min-width: 2px; }\n'
        'canvas { border: 1px solid #2a2a4a; border-radius: 8px; cursor: crosshair; }\n'
        '.tooltip { position: fixed; background: #1a1a2e; border: 1px solid #00d4ff; padding: 8px 12px; border-radius: 6px; font-size: 12px; pointer-events: none; display: none; z-index: 100; }\n'
        '.tag { display: inline-block; background: #16213e; padding: 2px 8px; border-radius: 4px; margin: 2px; font-size: 12px; }\n'
        '.tree { padding-left: 20px; } .tree-item { padding: 4px 0; }\n'
        '.type { color: #c792ea; font-size: 12px; } .shape { color: #82aaff; font-family: monospace; }\n'
        '#search { width: 100%; padding: 8px 12px; background: #16213e; border: 1px solid #2a2a4a; border-radius: 8px; color: #e0e0e0; font-size: 14px; margin-bottom: 15px; }\n'
        '#search:focus { outline: none; border-color: #00d4ff; }\n'
        '.tab-bar { display: flex; gap: 4px; margin-bottom: 15px; }\n'
        '.tab { padding: 8px 16px; background: #16213e; border: 1px solid #2a2a4a; border-radius: 8px 8px 0 0; cursor: pointer; }\n'
        '.tab.active { background: #1a1a2e; border-bottom-color: #1a1a2e; color: #00d4ff; }\n'
        '.tab-content { display: none; } .tab-content.active { display: block; }\n'
        '</style>\n</head>\n<body>\n'
        '<h1>SAT .pt File Viewer</h1>\n'
        '<div class="tab-bar">\n'
        '  <div class="tab active" onclick="switchTab(\'overview\')">Overview</div>\n'
        '  <div class="tab" onclick="switchTab(\'embedding\')">Entity Embeddings</div>\n'
        '  <div class="tab" onclick="switchTab(\'layers\')">Model Layers</div>\n'
        '  <div class="tab" onclick="switchTab(\'structure\')">Data Structure</div>\n'
        '</div>\n'
        '<div id="tab-overview" class="tab-content active">\n'
        '<h2>文件概览</h2><div class="grid">\n' + overview_cards + '</div></div>\n'
        '<div id="tab-embedding" class="tab-content">\n'
        '<h2>Entity Embeddings PCA 可视化</h2>\n'
        '<p style="color:#888;margin-bottom:10px;">' + str(n_points) + ' 个实体投影到 2D (PCA explained variance: ' + exp0 + ' + ' + exp1 + ') — 鼠标悬停查看实体 ID</p>\n'
        '<canvas id="scatter" width="900" height="600"></canvas>\n'
        '<div class="tooltip" id="tooltip"></div></div>\n'
        '<div id="tab-layers" class="tab-content">\n'
        '<h2>模型参数层级 (aligner_best.pkl)</h2>\n'
        '<input type="text" id="search" placeholder="搜索层名... (如 transformer, gnn, embedding)" oninput="filterLayers()">\n'
        '<table id="layerTable"><thead><tr><th>层名</th><th>形状</th><th>参数量</th><th>大小(MB)</th><th>占比</th></tr></thead><tbody>\n'
        + layer_rows +
        '</tbody></table>\n'
        '<div style="margin-top:10px;color:#888;">总参数量: ' + f"{total_params:,}" + ' | 总大小: ' + f"{total_mb:.2f}" + ' MB</div>\n'
        '</div>\n'
        '<div id="tab-structure" class="tab-content"><h2>数据结构树</h2><div id="structTree"></div></div>\n'
        '<script>\n'
        'const embData = ' + emb_json + ';\n'
        'const fileStructures = ' + struct_json + ';\n'
        'const fileNames = ' + names_json + ';\n'
        'function switchTab(name) {\n'
        '  document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));\n'
        '  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));\n'
        '  document.getElementById("tab-"+name).classList.add("active");\n'
        '  event.target.classList.add("active");\n'
        '  if (name === "embedding") drawScatter();\n'
        '  if (name === "structure") drawTree();\n'
        '}\n'
        'let scatterDrawn = false;\n'
        'function drawScatter() {\n'
        '  if (scatterDrawn) return; scatterDrawn = true;\n'
        '  const canvas = document.getElementById("scatter"), ctx = canvas.getContext("2d");\n'
        '  const pts = embData.coords, ids = embData.indices;\n'
        '  const W = canvas.width, H = canvas.height, pad = 40;\n'
        '  let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;\n'
        '  pts.forEach(p => { minX=Math.min(minX,p[0]); maxX=Math.max(maxX,p[0]); minY=Math.min(minY,p[1]); maxY=Math.max(maxY,p[1]); });\n'
        '  const sx=(W-2*pad)/(maxX-minX||1), sy=(H-2*pad)/(maxY-minY||1);\n'
        '  const tx=x=>pad+(x-minX)*sx, ty=y=>H-pad-(y-minY)*sy;\n'
        '  ctx.fillStyle="#0f0f23"; ctx.fillRect(0,0,W,H);\n'
        '  pts.forEach((p,i)=>{ const hue=(ids[i]*137.5)%360; ctx.fillStyle=`hsla(${hue},70%,60%,0.6)`; ctx.beginPath(); ctx.arc(tx(p[0]),ty(p[1]),3,0,Math.PI*2); ctx.fill(); });\n'
        '  const tooltip=document.getElementById("tooltip");\n'
        '  canvas.addEventListener("mousemove",e=>{\n'
        '    const rect=canvas.getBoundingClientRect(), mx=e.clientX-rect.left, my=e.clientY-rect.top;\n'
        '    let closest=-1, minD=100;\n'
        '    pts.forEach((p,i)=>{ const d=Math.hypot(tx(p[0])-mx,ty(p[1])-my); if(d<minD){minD=d;closest=i;} });\n'
        '    if(closest>=0&&minD<15){ tooltip.style.display="block"; tooltip.style.left=(e.clientX+12)+"px"; tooltip.style.top=(e.clientY+12)+"px";\n'
        '      tooltip.innerHTML=`Entity ID: <b style="color:#00d4ff">${ids[closest]}</b><br>x: ${pts[closest][0].toFixed(3)}, y: ${pts[closest][1].toFixed(3)}`;\n'
        '    } else { tooltip.style.display="none"; }\n'
        '  });\n'
        '}\n'
        'function filterLayers() {\n'
        '  const q=document.getElementById("search").value.toLowerCase();\n'
        '  document.querySelectorAll(".layer-row").forEach(row=>{ row.style.display=row.dataset.name.includes(q)?"":"none"; });\n'
        '}\n'
        'let treeDrawn=false;\n'
        'function drawTree() {\n'
        '  if(treeDrawn) return; treeDrawn=true;\n'
        '  const container=document.getElementById("structTree"); let html="";\n'
        '  fileStructures.forEach((struct,i)=>{ html+=`<div class="card" style="margin-bottom:15px"><h3>${fileNames[i]}</h3>`+renderNode(struct)+"</div>"; });\n'
        '  container.innerHTML=html;\n'
        '}\n'
        'function renderNode(node,depth=0) {\n'
        '  if(depth>5) return \'<span style="color:#666">...</span>\';\n'
        '  let html=\'<div class="tree-item">\';\n'
        '  html+=`<span class="type">${node.type}</span> `;\n'
        '  if(node.shape) html+=`<span class="shape">${JSON.stringify(node.shape)}</span> `;\n'
        '  if(node.dtype) html+=`<span class="tag">${node.dtype}</span> `;\n'
        '  if(node.numel) html+=`<span class="tag">${node.numel.toLocaleString()} elements</span> `;\n'
        '  if(node.min!==undefined) html+=`<span class="tag">range: [${node.min.toFixed(3)}, ${node.max.toFixed(3)}]</span> `;\n'
        '  if(node.keys) html+=node.keys.map(k=>`<span class="tag">${k}</span>`).join(" ");\n'
        '  if(node.value) html+=`<span style="color:#c3e88d">${node.value}</span>`;\n'
        '  if(node.children){ html+=\'<div class="tree">\'; node.children.forEach(c=>{ html+=`<div><b style="color:#82aaff">${c.name}</b>: `+renderNode(c,depth+1)+"</div>"; }); html+="</div>"; }\n'
        '  html+="</div>"; return html;\n'
        '}\n'
        '</script>\n</body>\n</html>'
    )


def main():
    files = [
        ("entity_embedding.pt", "实体嵌入矩阵"),
        ("graph_data_all.pt", "图结构数据"),
        ("aligner_best.pkl", "Aligner 模型权重"),
        ("config.json", "模型配置"),
    ]

    file_infos = []
    embedding_tensor = None
    layer_data = []

    for fname, desc in files:
        path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(path):
            continue
        size_bytes = os.path.getsize(path)
        if size_bytes > 1024 * 1024:
            size_str = f"{size_bytes / 1024 / 1024:.1f} MB"
        else:
            size_str = f"{size_bytes / 1024:.1f} KB"

        if fname.endswith('.json'):
            with open(path) as f:
                obj = json.load(f)
            struct = describe(obj, fname)
            file_infos.append({
                "name": fname, "size": size_str, "top_type": "JSON Config",
                "summary": f"{len(obj)} keys", "struct": struct
            })
            continue

        obj, struct = inspect_pt(path)

        if fname == "entity_embedding.pt":
            embedding_tensor = obj
            summary = f"Tensor {list(obj.shape)}, {obj.dtype}"
            top_type = "Tensor"
        elif fname == "graph_data_all.pt":
            summary = f"Dict with keys: {list(obj.keys())}"
            top_type = "Dict[str, Data]"
        elif fname == "aligner_best.pkl":
            if isinstance(obj, dict):
                layers = state_dict_layer_sizes(obj)
                max_params = max(l['params'] for l in layers)
                for l in layers:
                    l['pct'] = round(l['params'] / max_params * 100, 1)
                layer_data = layers
                summary = f"{len(obj)} layers, {sum(l['params'] for l in layers):,} params"
            top_type = "State Dict"
        else:
            summary = type(obj).__name__
            top_type = type(obj).__name__

        file_infos.append({
            "name": fname, "size": size_str, "top_type": top_type,
            "summary": summary, "struct": struct
        })

    # PCA embedding
    if embedding_tensor is not None:
        coords, indices, explained = embedding_to_2d(embedding_tensor)
        emb_data = {"coords": coords, "indices": indices, "explained": explained,
                    "n_points": len(coords)}
    else:
        emb_data = {"coords": [], "indices": [], "explained": [0, 0], "n_points": 0}

    html = build_html(file_infos, emb_data, layer_data)
    out_path = "pt_viewer.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已生成: {os.path.abspath(out_path)}")
    print("用浏览器打开即可查看")

    os.system(f"open {out_path}")


if __name__ == "__main__":
    main()
