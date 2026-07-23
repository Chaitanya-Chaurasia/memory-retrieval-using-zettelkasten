export interface Note {
  id: number;
  content: string;
  context: string;
  keywords: string[];
  tags: string[];
  created_at: number;
  updated_at: number;
  access_count: number;
  last_accessed: number;
}

export interface NoteLink {
  a: number;
  b: number;
  reason: string;
}

export interface MemoryDump {
  notes: Note[];
  links: NoteLink[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

/** SSE events from the backend, plus the client-side "_thinking" aggregate. */
export interface SSEEvent {
  type: string;
  [key: string]: any;
}
