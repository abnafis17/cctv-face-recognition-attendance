export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:3001";

async function readError(res: Response) {
  const txt = await res.text();
  // try to make error readable even if backend returns JSON
  try {
    const j = JSON.parse(txt);
    return typeof j === "string" ? j : JSON.stringify(j);
  } catch {
    return txt;
  }
}

export async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<T>;
}

export async function postJSON<T>(path: string, body?: any): Promise<T> {
  const res = await fetch(`${BACKEND_URL}${path}`, {
    method: "POST",
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json() as Promise<T>;
}
