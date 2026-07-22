"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

const API = "http://localhost:8000";

const STAGE_LABELS = {
  retrieval: "1 · memory retrieval",
  answering: "2 · answering",
  memorizing: "3 · memorizing",
};

function Label({ children, className }) {
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

function Ev({ children, className, strong }) {
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

function TimelineEvent({ ev }) {
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
            hit · note {ev.id} · rrf {ev.score} · {ev.sources.join("+")}
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
    case "fact_extracted":
      return (
        <Ev>
          <Label>fact extracted</Label>
          {ev.fact}
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
            {ev.note.tags.map((t) => (
              <Badge key={t} variant="outline" className="font-mono text-[10px] font-normal">
                #{t}
              </Badge>
            ))}
            {ev.note.keywords.map((k) => (
              <Badge key={k} variant="secondary" className="font-mono text-[10px] font-normal">
                {k}
              </Badge>
            ))}
          </div>
        </Ev>
      );
    case "evolution_check":
      return (
        <Ev>
          <Label>checking neighbors</Label>
          note {ev.note_id} vs [{ev.neighbor_ids.join(", ")}]
        </Ev>
      );
    case "link_created":
      return (
        <Ev strong>
          <Label>
            link {ev.a} &harr; {ev.b}
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

function ThinkingBlock({ text, live }) {
  if (!text) return null;
  return (
    <Ev className="whitespace-pre-wrap font-mono text-[12px]">
      <Label>thinking{live ? "…" : ""}</Label>
      {text}
    </Ev>
  );
}

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [events, setEvents] = useState([]);
  const [mem, setMem] = useState({ notes: [], links: [] });
  const chatEnd = useRef(null);
  const tlEnd = useRef(null);

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
    const history = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "", streaming: true }]);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: history.map(({ role, content }) => ({ role, content })),
        }),
      });
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      const handle = (ev) => {
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
        buf = parts.pop();
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
        <header className="flex items-baseline gap-3 border-b px-6 py-3.5">
          <h1 className="text-sm font-semibold tracking-tight">amem</h1>
          <span className="font-mono text-[11px] text-muted-foreground">
            a-mem · opus 4.8 · sqlite
          </span>
        </header>

        <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-6 py-5">
          {messages.length === 0 && (
            <p className="m-auto max-w-xs text-center text-sm text-muted-foreground">
              Say anything. Facts worth keeping become linked notes in a local SQLite memory.
            </p>
          )}
          {messages.map((m, i) => (
            <div
              key={i}
              className={cn(
                "max-w-[82%] whitespace-pre-wrap rounded-lg px-3.5 py-2 text-sm",
                m.role === "user"
                  ? "self-end bg-foreground text-background"
                  : "self-start border bg-background",
                m.streaming && busy && "caret"
              )}
            >
              {m.content}
            </div>
          ))}
          <div ref={chatEnd} />
        </div>

        <div className="flex gap-2 border-t px-6 py-4">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Say anything…"
            disabled={busy}
          />
          <Button onClick={send} disabled={busy || !input.trim()}>
            Send
          </Button>
        </div>
      </div>

      {/* side panel */}
      <div className="flex min-w-0 flex-1 flex-col">
        <Tabs defaultValue="timeline" className="flex h-full flex-col gap-0">
          <div className="border-b px-4 py-2">
            <TabsList className="h-8">
              <TabsTrigger value="timeline" className="text-xs">
                Timeline
              </TabsTrigger>
              <TabsTrigger value="memories" className="text-xs" onClick={refreshMemories}>
                Memories ({mem.notes.length})
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
        </Tabs>
      </div>
    </div>
  );
}
