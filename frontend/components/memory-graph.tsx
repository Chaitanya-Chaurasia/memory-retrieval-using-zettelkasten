"use client";

import { useEffect, useMemo, useState } from "react";
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  type SimulationNodeDatum,
} from "d3-force";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Note, NoteLink } from "@/lib/types";

const W = 640;
const H = 480;

type GraphNode = Note & SimulationNodeDatum;
interface GraphEdge {
  source: GraphNode;
  target: GraphNode;
  reason: string;
}

export function MemoryGraph({ notes, links }: { notes: Note[]; links: NoteLink[] }) {
  const [layout, setLayout] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] }>({
    nodes: [],
    edges: [],
  });
  const [selected, setSelected] = useState<number | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null);

  useEffect(() => {
    if (notes.length === 0) {
      setLayout({ nodes: [], edges: [] });
      return;
    }
    const nodes: GraphNode[] = notes.map((n) => ({ ...n }));
    const edges = links.map((l) => ({ source: l.a, target: l.b, reason: l.reason }));
    const sim = forceSimulation(nodes)
      .force(
        "link",
        forceLink(edges)
          .id((d) => (d as GraphNode).id)
          .distance(80)
          .strength(0.6)
      )
      .force("charge", forceManyBody().strength(-160))
      .force("center", forceCenter(W / 2, H / 2))
      .force("collide", forceCollide(26))
      .stop();
    sim.tick(300);
    setLayout({ nodes, edges: edges as unknown as GraphEdge[] });
  }, [notes, links]);

  const selectedNote = useMemo(
    () => layout.nodes.find((n) => n.id === selected),
    [layout, selected]
  );
  const radius = (n: GraphNode) => 7 + Math.sqrt(n.access_count || 0) * 2.5;

  if (notes.length === 0) {
    return (
      <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-muted-foreground">
        no notes yet
      </p>
    );
  }

  return (
    <div className="flex h-full flex-col gap-3">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full flex-1 rounded-lg border"
        onClick={() => {
          setSelected(null);
          setSelectedEdge(null);
        }}
      >
        {layout.edges.map((e, i) => (
          <g key={i}>
            <line
              x1={e.source.x}
              y1={e.source.y}
              x2={e.target.x}
              y2={e.target.y}
              className={cn(
                "stroke-border",
                (selectedEdge === e ||
                  (selected != null &&
                    (e.source.id === selected || e.target.id === selected))) &&
                  "stroke-foreground"
              )}
              strokeWidth="1.2"
            />
            <line
              x1={e.source.x}
              y1={e.source.y}
              x2={e.target.x}
              y2={e.target.y}
              stroke="transparent"
              strokeWidth="10"
              className="cursor-pointer"
              onClick={(ev) => {
                ev.stopPropagation();
                setSelected(null);
                setSelectedEdge(selectedEdge === e ? null : e);
              }}
            />
          </g>
        ))}
        {layout.nodes.map((n) => (
          <g
            key={n.id}
            transform={`translate(${n.x},${n.y})`}
            className="cursor-pointer"
            onClick={(ev) => {
              ev.stopPropagation();
              setSelectedEdge(null);
              setSelected(n.id === selected ? null : n.id);
            }}
          >
            <circle r={radius(n) + 6} fill="transparent" />
            <circle
              r={radius(n)}
              className={cn(
                "stroke-foreground",
                n.id === selected ? "fill-foreground" : "fill-background"
              )}
              strokeWidth="1.3"
            />
            <text
              y={radius(n) + 11}
              textAnchor="middle"
              className="fill-muted-foreground font-mono text-[9px]"
            >
              {n.id}
            </text>
          </g>
        ))}
      </svg>

      {selectedEdge ? (
        <div className="rounded-lg border p-3">
          <span className="block font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
            link {selectedEdge.source.id} &harr; {selectedEdge.target.id}
          </span>
          <p className="mt-1 text-[13px]">{selectedEdge.reason || "(no reason recorded)"}</p>
        </div>
      ) : selectedNote ? (
        <div className="rounded-lg border p-3">
          <span className="block font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
            note {selectedNote.id} · recalled {selectedNote.access_count}&times;
          </span>
          <p className="mt-1 text-[13px] font-medium">{selectedNote.content}</p>
          <p className="text-[12px] italic text-muted-foreground">{selectedNote.context}</p>
          <div className="mt-1.5 flex flex-wrap gap-1">
            {(selectedNote.tags ?? []).map((t) => (
              <Badge key={t} variant="outline" className="font-mono text-[10px] font-normal">
                #{t}
              </Badge>
            ))}
          </div>
        </div>
      ) : (
        <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
          node size = recall count · click a node or an edge for details
        </p>
      )}
    </div>
  );
}
