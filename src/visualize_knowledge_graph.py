"""
Script de Visualización del Grafo de Conocimiento Médico (Graph RAG)
Genera:
1. Visualización interactiva en HTML con simulación de fuerzas ForceAtlas2 (Vis.js).
2. Gráfico estático de alta resolución usando el algoritmo Fruchterman-Reingold (NetworkX + Matplotlib).
"""

import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.medical_knowledge_graph import obtener_grafo_medico
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'knowledge_graph')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_MAP = {
    'Disease': '#3b82f6',      # Azul
    'Precaution': '#8b5cf6',   # Púrpura
    'Medication': '#ec4899',   # Rosa
    'Diet': '#10b981',         # Verde esmeralda
    'Severity': '#ef4444'      # Rojo
}

BORDER_MAP = {
    'Disease': '#1d4ed8',
    'Precaution': '#6d28d9',
    'Medication': '#be185d',
    'Diet': '#047857',
    'Severity': '#b91c1c'
}

def generar_visualizacion_fruchterman_reingold(G: nx.DiGraph, output_png: str):
    """
    Genera una figura estática con el algoritmo Fruchterman-Reingold (spring_layout).
    """
    print(f"Calculando disposición Fruchterman-Reingold para {G.number_of_nodes()} nodos...")
    plt.figure(figsize=(18, 14), facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#0f172a')

    # Layout de resortes (Fruchterman-Reingold)
    pos = nx.spring_layout(G, k=0.35, iterations=80, seed=42)

    # Agrupar nodos por tipo
    nodos_por_tipo = {}
    for node, data in G.nodes(data=True):
        ntype = data.get('type', 'Other')
        nodos_por_tipo.setdefault(ntype, []).append(node)

    # Dibujar aristas con transparencia
    edge_colors = []
    for u, v, data in G.edges(data=True):
        rel = data.get('relation', '')
        if rel == 'HAS_TRIAGE_SEVERITY':
            edge_colors.append('#f87171')
        elif rel == 'REQUIRES_PRECAUTION':
            edge_colors.append('#c084fc')
        elif rel == 'RECOMMENDS_MEDICATION':
            edge_colors.append('#f472b6')
        elif rel == 'SUGGESTS_DIET':
            edge_colors.append('#34d399')
        else:
            edge_colors.append('#94a3b8')

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        alpha=0.35,
        arrows=True,
        arrowsize=7,
        width=0.8,
        ax=ax
    )

    # Dibujar nodos por tipo
    for ntype, nodes in nodos_por_tipo.items():
        color = COLOR_MAP.get(ntype, '#94a3b8')
        size = 380 if ntype in ['Disease', 'Severity'] else 180
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=color,
            node_size=size,
            alpha=0.88,
            edgecolors='white',
            linewidths=0.6,
            ax=ax,
            label=f"{ntype} ({len(nodes)})"
        )

    # Etiquetas para enfermedades y severidad
    labels_to_draw = {}
    for node, data in G.nodes(data=True):
        ntype = data.get('type', '')
        if ntype in ['Disease', 'Severity']:
            # Abreviar si es largo
            label = str(node).replace('Severidad_', 'Nivel: ')
            if len(label) > 18:
                label = label[:16] + '..'
            labels_to_draw[node] = label

    nx.draw_networkx_labels(
        G, pos,
        labels=labels_to_draw,
        font_size=8,
        font_family='sans-serif',
        font_color='#ffffff',
        font_weight='bold',
        ax=ax
    )

    plt.title("Grafo de Conocimiento Médico (Graph RAG) — Disposición Fruchterman-Reingold",
              fontsize=16, color='#f8fafc', fontweight='bold', pad=20)
    
    legend = plt.legend(loc='upper right', frameon=True, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    print(f"Imagen guardada en: {output_png}")


def generar_visualizacion_interactiva_forceatlas2(G: nx.DiGraph, output_html: str):
    """
    Genera un archivo HTML interactivo con Vis.js utilizando el algoritmo de física ForceAtlas2.
    """
    print("Generando HTML interactivo con simulación ForceAtlas2...")
    
    vis_nodes = []
    for node, data in G.nodes(data=True):
        ntype = data.get('type', 'Other')
        bg_color = COLOR_MAP.get(ntype, '#94a3b8')
        border_color = BORDER_MAP.get(ntype, '#475569')
        
        desc = data.get('description') or data.get('text') or data.get('name') or data.get('clinical_action') or ''
        label = str(node)
        if label.startswith('Precaucion: '):
            label = label.replace('Precaucion: ', '')
        elif label.startswith('Farmaco: '):
            label = label.replace('Farmaco: ', '')
        elif label.startswith('Dieta: '):
            label = label.replace('Dieta: ', '')
        elif label.startswith('Severidad_'):
            label = 'TRIAGE: ' + label.replace('Severidad_', '')

        size = 28 if ntype in ['Disease', 'Severity'] else 16
        
        vis_nodes.append({
            'id': str(node),
            'label': label,
            'title': f"<b>Tipo:</b> {ntype}<br><b>Nodo:</b> {label}<br><b>Detalle:</b> {desc}",
            'group': ntype,
            'color': {
                'background': bg_color,
                'border': border_color,
                'highlight': {'background': '#fbbf24', 'border': '#d97706'}
            },
            'size': size,
            'font': {'color': '#1e293b', 'size': 12, 'face': 'Inter, sans-serif'}
        })

    vis_edges = []
    for u, v, data in G.edges(data=True):
        rel = data.get('relation', '')
        edge_color = '#94a3b8'
        width = 1.0
        dashes = False
        
        if rel == 'HAS_TRIAGE_SEVERITY':
            edge_color = '#ef4444'
            width = 2.5
            dashes = True
        elif rel == 'REQUIRES_PRECAUTION':
            edge_color = '#8b5cf6'
            width = 1.2
        elif rel == 'RECOMMENDS_MEDICATION':
            edge_color = '#ec4899'
            width = 1.2
        elif rel == 'SUGGESTS_DIET':
            edge_color = '#10b981'
            width = 1.2

        vis_edges.append({
            'from': str(u),
            'to': str(v),
            'label': rel,
            'title': f"Relación: {rel}",
            'color': {'color': edge_color, 'highlight': '#f59e0b'},
            'arrows': 'to',
            'width': width,
            'dashes': dashes,
            'font': {'size': 9, 'color': '#64748b', 'align': 'middle'}
        })

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Grafo de Conocimiento Médico (Graph RAG) — ForceAtlas2</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background-color: #0f172a;
      color: #f8fafc;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }}
    header {{
      background: #1e293b;
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #334155;
      box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
      z-index: 10;
    }}
    h1 {{ font-size: 1.2rem; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 8px; }}
    .subtitle {{ font-size: 0.85rem; color: #94a3b8; margin-top: 2px; }}
    .controls {{
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .search-box {{
      padding: 6px 12px;
      border-radius: 6px;
      border: 1px solid #475569;
      background: #0f172a;
      color: #f8fafc;
      font-size: 0.85rem;
      outline: none;
      width: 220px;
    }}
    .btn {{
      background: #2563eb;
      color: white;
      border: none;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      transition: background 0.2s;
    }}
    .btn:hover {{ background: #1d4ed8; }}
    #network-container {{
      flex: 1;
      width: 100%;
      height: 100%;
      position: relative;
    }}
    .legend {{
      position: absolute;
      bottom: 24px;
      left: 24px;
      background: rgba(30, 41, 59, 0.92);
      backdrop-filter: blur(8px);
      padding: 14px 18px;
      border-radius: 8px;
      border: 1px solid #334155;
      box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);
      font-size: 0.82rem;
      z-index: 5;
    }}
    .legend-title {{ font-weight: 700; margin-bottom: 8px; color: #e2e8f0; }}
    .legend-item {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
    .legend-color {{ width: 14px; height: 14px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.4); }}
    .info-panel {{
      position: absolute;
      top: 24px;
      right: 24px;
      width: 320px;
      background: rgba(30, 41, 59, 0.95);
      backdrop-filter: blur(8px);
      padding: 16px;
      border-radius: 8px;
      border: 1px solid #334155;
      box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5);
      font-size: 0.85rem;
      z-index: 5;
      display: none;
    }}
    .info-panel h3 {{ color: #38bdf8; margin-bottom: 8px; font-size: 1rem; }}
    .info-panel p {{ margin-bottom: 6px; color: #cbd5e1; line-height: 1.4; }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
      margin-bottom: 8px;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Ontología de Conocimiento Médico (Graph RAG)</h1>
      <div class="subtitle">Disposición ForceAtlas2 • {len(vis_nodes)} Nodos • {len(vis_edges)} Aristas Clínicas</div>
    </div>
    <div class="controls">
      <input type="text" id="search-input" class="search-box" placeholder="Buscar enfermedad o nodo...">
      <button class="btn" onclick="buscarNodo()">Buscar</button>
      <button class="btn" style="background:#475569;" onclick="network.fit({{animation: true}})">Centrar</button>
      <button class="btn" style="background:#059669;" id="physics-toggle" onclick="togglePhysics()">Pausar Física</button>
    </div>
  </header>

  <div id="network-container">
    <div class="legend">
      <div class="legend-title">Entidades Ontológicas:</div>
      <div class="legend-item"><span class="legend-color" style="background:{COLOR_MAP['Disease']}"></span> Disease (Enfermedad)</div>
      <div class="legend-item"><span class="legend-color" style="background:{COLOR_MAP['Precaution']}"></span> Precaution (Precaución)</div>
      <div class="legend-item"><span class="legend-color" style="background:{COLOR_MAP['Medication']}"></span> Medication (Fármaco)</div>
      <div class="legend-item"><span class="legend-color" style="background:{COLOR_MAP['Diet']}"></span> Diet (Dieta/Nutrición)</div>
      <div class="legend-item"><span class="legend-color" style="background:{COLOR_MAP['Severity']}"></span> Severity (Nivel Triaje)</div>
    </div>

    <div id="info-panel" class="info-panel">
      <h3 id="info-title">Detalle del Nodo</h3>
      <div id="info-badge"></div>
      <p><b>ID:</b> <span id="info-id"></span></p>
      <p><b>Descripción / Acción:</b> <span id="info-desc"></span></p>
      <p><b>Conexiones directas:</b> <span id="info-edges"></span></p>
    </div>
  </div>

  <script>
    const nodesData = {json.dumps(vis_nodes, ensure_ascii=False)};
    const edgesData = {json.dumps(vis_edges, ensure_ascii=False)};

    const container = document.getElementById('network-container');
    const nodes = new vis.DataSet(nodesData);
    const edges = new vis.DataSet(edgesData);

    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      nodes: {{
        shape: 'dot',
        scaling: {{ min: 14, max: 32 }},
        shadow: {{ enabled: true, color: 'rgba(0,0,0,0.5)', size: 8 }}
      }},
      edges: {{
        smooth: {{ type: 'continuous', roundness: 0.2 }},
        selectionWidth: 2
      }},
      physics: {{
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {{
          gravitationalConstant: -38,
          centralGravity: 0.008,
          springLength: 95,
          springConstant: 0.09,
          damping: 0.45,
          avoidOverlap: 0.4
        }},
        stabilization: {{ iterations: 120, updateInterval: 25 }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 100,
        navigationButtons: true,
        keyboard: true
      }}
    }};

    const network = new vis.Network(container, data, options);
    let physicsActive = true;

    function togglePhysics() {{
      physicsActive = !physicsActive;
      network.setOptions({{ physics: {{ enabled: physicsActive }} }});
      document.getElementById('physics-toggle').innerText = physicsActive ? 'Pausar Física' : 'Activar Física';
      document.getElementById('physics-toggle').style.background = physicsActive ? '#059669' : '#d97706';
    }}

    network.on('click', function(params) {{
      if (params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        const node = nodes.get(nodeId);
        const connectedEdges = network.getConnectedEdges(nodeId);

        document.getElementById('info-panel').style.display = 'block';
        document.getElementById('info-title').innerText = node.label;
        document.getElementById('info-id').innerText = node.id;
        document.getElementById('info-badge').innerHTML = `<span class="badge" style="background:${{node.color.background}}; color:#fff">${{node.group}}</span>`;
        document.getElementById('info-desc').innerHTML = node.title.split('<b>Detalle:</b> ')[1] || 'Sin información adicional.';
        document.getElementById('info-edges').innerText = connectedEdges.length + ' relaciones';
      }} else {{
        document.getElementById('info-panel').style.display = 'none';
      }}
    }});

    function buscarNodo() {{
      const query = document.getElementById('search-input').value.toLowerCase().trim();
      if (!query) return;

      const matches = nodes.get({{
        filter: function(item) {{
          return item.label.toLowerCase().includes(query) || item.id.toLowerCase().includes(query);
        }}
      }});

      if (matches.length > 0) {{
        const target = matches[0];
        network.focus(target.id, {{
          scale: 1.4,
          animation: {{ duration: 1000, easingFunction: 'easeInOutQuad' }}
        }});
        network.selectNodes([target.id]);
        document.getElementById('info-panel').style.display = 'block';
        document.getElementById('info-title').innerText = target.label;
        document.getElementById('info-id').innerText = target.id;
        document.getElementById('info-badge').innerHTML = `<span class="badge" style="background:${{target.color.background}}; color:#fff">${{target.group}}</span>`;
        document.getElementById('info-desc').innerHTML = target.title.split('<b>Detalle:</b> ')[1] || 'Sin información adicional.';
      }} else {{
        alert('No se encontró ningún nodo que coincida con: ' + query);
      }}
    }}

    document.getElementById('search-input').addEventListener('keypress', function(e) {{
      if (e.key === 'Enter') buscarNodo();
    }});
  </script>
</body>
</html>
"""
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Visualización interactiva guardada en: {output_html}")


if __name__ == '__main__':
    kg = obtener_grafo_medico()
    G = kg.graph
    
    png_path = os.path.join(OUTPUT_DIR, 'medical_knowledge_graph_fruchterman_reingold.png')
    html_path = os.path.join(OUTPUT_DIR, 'medical_knowledge_graph_interactive.html')
    
    generar_visualizacion_fruchterman_reingold(G, png_path)
    generar_visualizacion_interactiva_forceatlas2(G, html_path)
    print("\nVisualizaciones de grafos completadas exitosamente.")
