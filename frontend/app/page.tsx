"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { MemoryGraph } from "@/components/memory-graph";
import type { ChatMessage, MemoryDump, SSEEvent } from "@/lib/types";

const API = "http://localhost:8000";

const STAGE_LABELS: Record<string, string> = {
  retrieval: "1 · memory retrieval",
  answering: "2 · answering",
  memorizing: "3 · memorizing",
};

function Label({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span
      className={cn(
        "block font-mono text-[10px] uppercase tracking-[0.12em] text-muted-foreground",
        className
      )}
    >
      {children}
    </span>
  );
}

function Ev({
  children,
  className,
  strong,
}: {
  children: ReactNode;
  className?: string;
  strong?: boolean;
}) {
  return (
    <div
      className={cn(
        "border-l pl-3 py-1 text-[13px] leading-relaxed text-muted-foreground",
        strong ? "border-foreground" : "border-border",
        className
      )}
    >
      {children}
    </div>
  );
}

function Bar({ frac }: { frac: number }) {
  return (
    <span className="inline-block h-[6px] w-16 rounded-sm bg-muted align-middle">
      <span
        className="block h-full rounded-sm bg-foreground"
        style={{ width: `${Math.max(2, Math.min(100, frac * 100))}%` }}
      />
    </span>
  );
}

function RankerResults({ ev }: { ev: SSEEvent }) {
  const isVec = ev.ranker === "vector";
  const hits: SSEEvent[] = ev.hits ?? [];
  const mag = (h: SSEEvent) => (isVec ? Math.max(0, (2 - h.distance) / 2) : Math.abs(h.score));
  const max = Math.max(...hits.map(mag), 1e-9);
  return (
    <Ev>
      <Label>
        {isVec ? "vector knn (cosine)" : "bm25 (fts5)"} · {hits.length} hits · {ev.ms}ms
      </Label>
      {hits.length === 0 && <div className="font-mono text-[11px]">no matches</div>}
      {hits.map((h, i) => (
        <div key={h.id} className="flex items-center gap-2 font-mono text-[11px]">
          <span className="w-8 text-right">#{i + 1}</span>
          <Bar frac={mag(h) / max} />
          <span className="w-14 shrink-0">
            {isVec ? `d=${h.distance}` : `s=${h.score}`}
          </span>
          <span className="truncate text-foreground">n{h.id} · {h.snippet}</span>
        </div>
      ))}
    </Ev>
  );
}

function RrfTable({ rows = [] }: { rows?: SSEEvent[] }) {
  const max = Math.max(...rows.map((r) => r.total), 1e-9);
  return (
    <Ev strong>
      <Label>rrf fusion · score = Σ 1/(60+rank)</Label>
      {rows.map((r) => (
        <div
          key={r.id}
          className={cn(
            "flex items-center gap-2 font-mono text-[11px]",
            !r.selected && "opacity-40"
          )}
        >
          <span className="w-8 shrink-0 text-right">n{r.id}</span>
          <Bar frac={r.total / max} />
          <span className="w-40 shrink-0">
            {r.vec_contrib > 0 ? `vec@${r.vec_rank + 1}=${r.vec_contrib}` : "vec —"}
            {" + "}
            {r.bm25_contrib > 0 ? `bm25@${r.bm25_rank + 1}=${r.bm25_contrib}` : "bm25 —"}
          </span>
          <span className="w-16 shrink-0 text-foreground">= {r.total}</span>
          <span className="truncate">{r.selected ? r.snippet : "cut"}</span>
        </div>
      ))}
    </Ev>
  );
}

function RerankTable({ rows = [] }: { rows?: SSEEvent[] }) {
  const max = Math.max(...rows.map((r) => r.final), 1e-9);
  return (
    <Ev strong>
      <Label>memorybank rerank · 0.75·rel + 0.25·e^(−age/S)</Label>
      {rows.map((r) => (
        <div
          key={r.id}
          className={cn(
            "flex items-center gap-2 font-mono text-[11px]",
            !r.selected && "opacity-40"
          )}
        >
          <span className="w-8 shrink-0 text-right">n{r.id}</span>
          <Bar frac={r.final / max} />
          <span className="w-44 shrink-0">
            rel={r.rel} mem={r.mem} → {r.final}
          </span>
          <span className="w-24 shrink-0">
            {r.age_h}h · {r.recalls}×
          </span>
          <span className="truncate">{r.selected ? r.snippet : "cut"}</span>
        </div>
      ))}
    </Ev>
  );
}

function TimelineEvent({ ev }: { ev: SSEEvent }) {
  switch (ev.type) {
    case "stage":
      return (
        <Ev strong>
          <Label className="text-foreground">{STAGE_LABELS[ev.name] || ev.name}</Label>
        </Ev>
      );
    case "memory_search":
      return (
        <Ev>
          <Label>search</Label>
          <span className="text-foreground">{ev.method}</span> for &ldquo;{ev.query}&rdquo;
        </Ev>
      );
    case "memory_hit":
      return (
        <Ev strong>
          <Label>
            hit · note {ev.id} · score {ev.score} · {(ev.sources ?? []).join("+")}
          </Label>
          <span className="font-medium text-foreground">{ev.content}</span>
          <div>{ev.context}</div>
        </Ev>
      );
    case "memory_miss":
      return (
        <Ev>
          <Label>no memories matched</Label>
        </Ev>
      );
    case "embed":
      return (
        <Ev>
          <Label>embed</Label>
          <span className="font-mono text-[11px]">
            query &rarr; {ev.model} &rarr; float32[{ev.dims}]
          </span>
        </Ev>
      );
    case "ranker_results":
      return <RankerResults ev={ev} />;
    case "rrf_table":
      return <RrfTable rows={ev.rows} />;
    case "rerank_table":
      return <RerankTable rows={ev.rows} />;
    case "extracting_facts":
      return (
        <Ev>
          <Label>extracting facts · {ev.model}</Label>
          scanning the exchange for durable facts about you
        </Ev>
      );
    case "fact_extracted":
      return (
        <Ev>
          <Label>fact extracted</Label>
          {ev.fact}
        </Ev>
      );
    case "write_decision":
      return (
        <Ev
          strong={ev.action !== "noop"}
          className={ev.action === "noop" ? "opacity-50" : undefined}
        >
          <Label>
            {ev.action === "noop" && `noop · already known${ev.target ? ` (note ${ev.target})` : ""}`}
            {ev.action === "update" && `update · merging into note ${ev.target}`}
            {ev.action === "add" && "add · new information"}
          </Label>
          {ev.reason}
        </Ev>
      );
    case "note_updated":
      return (
        <Ev strong>
          <Label>note {ev.note.id} rewritten</Label>
          <span className="font-medium text-foreground">{ev.note.content}</span>
        </Ev>
      );
    case "no_facts":
      return (
        <Ev>
          <Label>nothing worth remembering</Label>
        </Ev>
      );
    case "note_created":
      return (
        <Ev strong>
          <Label>note {ev.note.id} created</Label>
          <span className="font-medium text-foreground">{ev.note.content}</span>
          <div>{ev.note.context}</div>
          <div className="mt-1.5 flex flex-wrap gap-1">
            {(ev.note.tags ?? []).map((t: string) => (
              <Badge key={t} variant="outline" className="font-mono text-[10px] font-normal">
                #{t}
              </Badge>
            ))}
            {(ev.note.keywords ?? []).map((k: string) => (
              <Badge key={k} variant="secondary" className="font-mono text-[10px] font-normal">
                {k}
              </Badge>
            ))}
          </div>
        </Ev>
      );
    case "evolution_check": {
      const neighbors: SSEEvent[] = ev.neighbors ?? [];
      const max = Math.max(...neighbors.map((n) => 2 - n.distance), 1e-9);
      return (
        <Ev>
          <Label>nearest neighbors of note {ev.note_id}</Label>
          {neighbors.map((n) => (
            <div key={n.id} className="flex items-center gap-2 font-mono text-[11px]">
              <span className="w-8 shrink-0 text-right">n{n.id}</span>
              <Bar frac={Math.max(0, 2 - n.distance) / max} />
              <span className="w-16 shrink-0">d={n.distance}</span>
              <span className="truncate">{n.snippet}</span>
            </div>
          ))}
        </Ev>
      );
    }
    case "link_decision":
      return (
        <Ev strong={ev.linked} className={!ev.linked ? "opacity-50" : undefined}>
          <Label>
            {ev.linked ? `link ${ev.a} ↔ ${ev.b}` : `no link ${ev.a} · ${ev.b}`}
          </Label>
          {ev.reason}
        </Ev>
      );
    case "note_evolved":
      return (
        <Ev strong>
          <Label>note {ev.note_id} evolved</Label>
          {ev.new_context}
        </Ev>
      );
    case "error":
      return (
        <Ev className="border-destructive">
          <Label className="text-destructive">error in {ev.where}</Label>
          {ev.message}
        </Ev>
      );
    default:
      return null;
  }
}

function ThinkingBlock({ text, live }: { text: string; live: boolean }) {
  if (!text) return null;
  return (
    <Ev className="whitespace-pre-wrap font-mono text-[12px]">
      <Label>thinking{live ? "…" : ""}</Label>
      {text}
    </Ev>
  );
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [mem, setMem] = useState<MemoryDump>({ notes: [], links: [] });
  const chatEnd = useRef<HTMLDivElement>(null);
  const tlEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  useEffect(() => {
    tlEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  async function refreshMemories() {
    try {
      const r = await fetch(`${API}/memories`);
      setMem(await r.json());
    } catch {}
  }
  useEffect(() => {
    refreshMemories();
  }, []);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    setEvents([]);
    const history: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "", streaming: true }]);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history.map(({ role, content }) => ({ role, content })),
        }),
      });
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      const handle = (ev: SSEEvent) => {
        if (ev.type === "token_delta") {
          setMessages((ms) => {
            const copy = [...ms];
            const last = copy[copy.length - 1];
            copy[copy.length - 1] = { ...last, content: last.content + ev.text };
            return copy;
          });
        } else if (ev.type === "thinking_delta") {
          setEvents((es) => {
            const copy = [...es];
            const last = copy[copy.length - 1];
            if (last && last.type === "_thinking") {
              copy[copy.length - 1] = { ...last, text: last.text + ev.text };
            } else {
              copy.push({ type: "_thinking", text: ev.text });
            }
            return copy;
          });
        } else if (ev.type === "done") {
          refreshMemories();
        } else {
          setEvents((es) => [...es, ev]);
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.trim();
          if (line.startsWith("data: ")) handle(JSON.parse(line.slice(6)));
        }
      }
    } catch (e) {
      setEvents((es) => [...es, { type: "error", where: "network", message: String(e) }]);
    } finally {
      setBusy(false);
      setMessages((ms) => ms.map((m) => ({ ...m, streaming: false })));
    }
  }

  return (
    <div className="flex h-screen">
      {/* chat */}
      <div className="flex min-w-0 flex-[1.2] flex-col border-r">
        <header className="flex items-baseline gap-3 px-6 py-3.5">
          <h1 className="text-sm font-semibold tracking-tight">
            long-term memory retrieval using zettelkasten
          </h1>
          <span className="ml-auto flex gap-3 text-[11px]">
            <a
              href="https://arxiv.org/abs/2502.12110"
              target="_blank"
              rel="noreferrer"
              className="text-muted-foreground underline underline-offset-2 hover:text-foreground"
            >
              a-mem paper
            </a>
            <a
              href="https://arxiv.org/abs/2504.19413"
              target="_blank"
              rel="noreferrer"
              className="text-muted-foreground underline underline-offset-2 hover:text-foreground"
            >
              mem0 paper
            </a>
          </span>
        </header>

        <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-6 py-5">
          {messages.length === 0 && (
            <p className="m-auto max-w-xs text-center text-sm text-muted-foreground">
              say anything. facts worth keeping become linked notes in a local sqlite memory.
            </p>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              className={cn(
                "max-w-[82%] rounded-2xl px-4 py-1.5 text-sm",
                m.role === "user"
                  ? "self-end whitespace-pre-wrap rounded-br-md bg-[#007AFF] text-white"
                  : "self-start rounded-bl-md border bg-white text-black",
                m.streaming && busy && "caret"
              )}
            >
              {m.role === "assistant" ? (
                <div className="prose prose-sm prose-neutral max-w-none prose-headings:tracking-tight prose-p:my-1 prose-p:first:mt-0 prose-p:last:mb-0 prose-pre:my-2 prose-pre:rounded-md prose-pre:bg-muted prose-pre:text-foreground prose-code:font-mono prose-code:text-[12px] prose-code:before:content-none prose-code:after:content-none prose-ul:my-1.5 prose-li:my-0">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                </div>
              ) : (
                m.content
              )}
            </div>
          ))}
          <div ref={chatEnd} />
        </div>

        <div className="flex gap-2 border-t px-6 py-4">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="say anything…"
            disabled={busy}
          />
          <Button onClick={send} disabled={busy || !input.trim()}>
            send
          </Button>
        </div>
      </div>

      {/* side panel */}
      <div className="flex min-w-0 flex-1 flex-col">
        <Tabs defaultValue="timeline" className="flex h-full flex-col gap-0">
          <div className="px-4 py-2">
            <TabsList className="h-8">
              <TabsTrigger value="timeline" className="text-xs">
                timeline
              </TabsTrigger>
              <TabsTrigger value="memories" className="text-xs" onClick={refreshMemories}>
                memories ({mem.notes.length})
              </TabsTrigger>
              <TabsTrigger value="graph" className="text-xs" onClick={refreshMemories}>
                graph
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="timeline" className="flex-1 overflow-y-auto p-4">
            <div className="flex flex-col gap-1.5">
              {events.length === 0 && (
                <Ev>
                  <Label>idle</Label>
                  send a message to watch the backend think
                </Ev>
              )}
              {events.map((ev, i) =>
                ev.type === "_thinking" ? (
                  <ThinkingBlock key={i} text={ev.text} live={busy && i === events.length - 1} />
                ) : (
                  <TimelineEvent key={i} ev={ev} />
                )
              )}
              <div ref={tlEnd} />
            </div>
          </TabsContent>

          <TabsContent value="memories" className="flex-1 overflow-y-auto p-4">
            <div className="flex flex-col gap-2">
              {mem.notes.length === 0 && (
                <Ev>
                  <Label>empty</Label>
                  no memories stored yet
                </Ev>
              )}
              {mem.notes.map((n) => {
                const linked = mem.links
                  .filter((l) => l.a === n.id || l.b === n.id)
                  .map((l) => (l.a === n.id ? l.b : l.a));
                return (
                  <div key={n.id} className="rounded-lg border p-3">
                    <Label>
                      note {n.id} · recalled {n.access_count}&times;
                      {linked.length > 0 && <> · linked [{linked.join(", ")}]</>}
                    </Label>
                    <p className="mt-1 text-[13px] font-medium">{n.content}</p>
                    <p className="text-[12px] italic text-muted-foreground">{n.context}</p>
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {n.tags.map((t) => (
                        <Badge
                          key={t}
                          variant="outline"
                          className="font-mono text-[10px] font-normal"
                        >
                          #{t}
                        </Badge>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </TabsContent>
          <TabsContent value="graph" className="flex-1 overflow-y-auto p-4">
            <MemoryGraph notes={mem.notes} links={mem.links} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
