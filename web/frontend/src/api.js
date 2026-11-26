const API_BASE = "/api";

async function parseResponse(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Lỗi máy chủ (HTTP ${res.status})`);
  }
  return data;
}

export async function fetchHealth() {
  return parseResponse(await fetch(`${API_BASE}/health`));
}

export async function fetchJobs() {
  return parseResponse(await fetch(`${API_BASE}/jobs`));
}

export async function fetchJob(jobId) {
  return parseResponse(await fetch(`${API_BASE}/jobs/${jobId}`));
}

export async function createJob({ title, description, file }) {
  if (file) {
    const form = new FormData();
    if (title) form.append("title", title);
    form.append("file", file);
    return parseResponse(await fetch(`${API_BASE}/jobs`, { method: "POST", body: form }));
  }
  return parseResponse(
    await fetch(`${API_BASE}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, description }),
    })
  );
}

export async function updateRequirements(jobId, payload) {
  return parseResponse(
    await fetch(`${API_BASE}/jobs/${jobId}/requirements`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

export async function uploadCvs(jobId, files) {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f);
  }
  return parseResponse(
    await fetch(`${API_BASE}/jobs/${jobId}/cvs`, {
      method: "POST",
      body: form,
    })
  );
}

export async function fetchJobCvs(jobId) {
  return parseResponse(await fetch(`${API_BASE}/jobs/${jobId}/cvs`));
}

export async function evaluateJob(jobId) {
  return parseResponse(
    await fetch(`${API_BASE}/jobs/${jobId}/evaluate`, { method: "POST" })
  );
}

export async function fetchRanking(jobId, level = null) {
  const qs = new URLSearchParams({ top_k: "100" });
  if (level !== null && level !== "all") {
    qs.set("level", String(level));
  }
  return parseResponse(await fetch(`${API_BASE}/jobs/${jobId}/ranking?${qs}`));
}

export async function fetchEvaluationDetail(evalId) {
  return parseResponse(await fetch(`${API_BASE}/evaluations/${evalId}`));
}

export async function deleteCandidate(candidateId) {
  return parseResponse(
    await fetch(`${API_BASE}/candidates/${candidateId}`, { method: "DELETE" })
  );
}
