// Pages serves static assets without Range support; Safari and iOS will not play
// a video unless the server answers byte ranges with 206. Slice the asset here.
export async function onRequestGet({ request, env }) {
  const range = request.headers.get("Range");
  const plain = new Request(new URL(request.url), { method: "GET" });
  const res = await env.ASSETS.fetch(plain);
  if (!res.ok) return res;
  const headers = new Headers(res.headers);
  headers.set("Accept-Ranges", "bytes");
  const m = range && /^bytes=(\d*)-(\d*)$/.exec(range.trim());
  if (!m) return new Response(res.body, { status: 200, headers });
  const body = await res.arrayBuffer();
  const size = body.byteLength;
  let start = m[1] === "" ? size - Number(m[2]) : Number(m[1]);
  let end = m[1] === "" || m[2] === "" ? size - 1 : Math.min(Number(m[2]), size - 1);
  if (!(start >= 0 && start <= end && start < size)) {
    headers.set("Content-Range", `bytes */${size}`);
    return new Response(null, { status: 416, headers });
  }
  headers.set("Content-Range", `bytes ${start}-${end}/${size}`);
  headers.set("Content-Length", String(end - start + 1));
  return new Response(body.slice(start, end + 1), { status: 206, headers });
}
