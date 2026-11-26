import React, { useEffect, useState } from "react";
import {
  createJob,
  deleteCandidate,
  evaluateJob,
  fetchEvaluationDetail,
  fetchHealth,
  fetchJob,
  fetchJobCvs,
  fetchJobs,
  fetchRanking,
  updateRequirements,
  uploadCvs,
} from "./api.js";

const SAMPLE_JD = {
  title: "Chuyên viên Phân tích Dữ liệu (Data Analyst)",
  description: `Vị trí: Chuyên viên Phân tích Dữ liệu (Data Analyst)
Yêu cầu bắt buộc:
- Tối thiểu 3 năm kinh nghiệm phân tích dữ liệu doanh nghiệp.
- Thành thạo Python, SQL, Excel và công cụ trực quan hóa Tableau hoặc Power BI.
- Kỹ năng mô hình hóa dữ liệu (data modeling) và lập báo cáo phân tích.
- Tốt nghiệp đại học chuyên ngành Khoa học Dữ liệu hoặc Công nghệ Thông tin.
Ưu tiên:
- Có kinh nghiệm làm việc với hệ thống ERP và tự động hóa bằng VBA.`,
};

const CRITERIA_LABELS = [
  ["sem", "Ngữ nghĩa văn bản"],
  ["skill", "Kỹ năng chuyên môn"],
  ["exp", "Số năm kinh nghiệm"],
  ["edu", "Học vấn và ngành"],
  ["cert", "Chứng chỉ nghề"],
];

function fmtScore(val) {
  if (val === null || val === undefined) return "Không yêu cầu";
  return Number(val).toFixed(1).replace(".", ",");
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [activeJob, setActiveJob] = useState(null);
  const [step, setStep] = useState(1);

  const [jdTitle, setJdTitle] = useState("");
  const [jdDesc, setJdDesc] = useState("");
  const [reqs, setReqs] = useState([]);
  const [reqYears, setReqYears] = useState(0);
  const [reqDegree, setReqDegree] = useState("none");
  const [newSkillText, setNewSkillText] = useState("");

  const [cvs, setCvs] = useState([]);
  const [uploadReport, setUploadReport] = useState([]);
  const [evalTimings, setEvalTimings] = useState(null);

  const [ranking, setRanking] = useState([]);
  const [levelFilter, setLevelFilter] = useState("all");
  const [detail, setDetail] = useState(null);

  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);

  async function refreshJobs(selectId = null) {
    const list = await fetchJobs();
    setJobs(list);
    if (selectId) {
      await openJob(selectId);
    }
  }

  async function openJob(jobId) {
    const job = await fetchJob(jobId);
    setActiveJob(job);
    setJdTitle(job.title);
    setJdDesc(job.description);
    setReqs(job.requirements || []);
    setReqYears(job.required_years || 0);
    setReqDegree(job.required_degree || "none");
    const cvList = await fetchJobCvs(jobId);
    setCvs(cvList);
    const rk = await fetchRanking(jobId);
    setRanking(rk.items || []);
    setDetail(null);
  }

  useEffect(() => {
    Promise.all([fetchHealth(), fetchJobs()])
      .then(([h, list]) => {
        setHealth(h);
        setJobs(list);
        if (list.length > 0) {
          openJob(list[0].id);
        }
      })
      .catch((err) => setNotice({ type: "error", text: err.message }));
  }, []);

  function startNewJob() {
    setActiveJob(null);
    setJdTitle("");
    setJdDesc("");
    setReqs([]);
    setReqYears(0);
    setReqDegree("none");
    setCvs([]);
    setUploadReport([]);
    setRanking([]);
    setDetail(null);
    setNotice(null);
    setStep(1);
  }

  async function handleCreateJob(e) {
    e.preventDefault();
    setBusy(true);
    setNotice(null);
    try {
      const created = await createJob({ title: jdTitle, description: jdDesc });
      await refreshJobs(created.id);
      setNotice({ type: "ok", text: `Đã phân tích và tạo vị trí "${created.title}". Bạn có thể rà soát lại các yêu cầu bắt buộc bên phải.` });
    } catch (err) {
      setNotice({ type: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  }

  async function handleSaveRequirements() {
    if (!activeJob) return;
    setBusy(true);
    setNotice(null);
    try {
      const updated = await updateRequirements(activeJob.id, {
        requirements: reqs.map((r) => ({
          text: r.text,
          type: r.type,
          is_mandatory: r.is_mandatory,
          weight: r.weight || 1.0,
        })),
        required_years: Number(reqYears) || 0,
        required_degree: reqDegree,
      });
      setActiveJob(updated);
      setReqs(updated.requirements || []);
      setNotice({ type: "ok", text: "Đã lưu danh sách yêu cầu bắt buộc và cập nhật bộ đệm véc-tơ JD." });
    } catch (err) {
      setNotice({ type: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  }

  function handleAddSkillReq() {
    const clean = newSkillText.trim().toLowerCase();
    if (!clean) return;
    setReqs([...reqs, { id: `tmp-${Date.now()}`, text: clean, type: "skill", is_mandatory: true, weight: 2.0 }]);
    setNewSkillText("");
  }

  async function handleFileSelect(fileList) {
    if (!activeJob || !fileList || fileList.length === 0) return;
    setBusy(true);
    setNotice(null);
    try {
      const res = await uploadCvs(activeJob.id, Array.from(fileList));
      setUploadReport(res.uploaded || []);
      setCvs(await fetchJobCvs(activeJob.id));
      await refreshJobs();
      const okCount = (res.uploaded || []).filter((x) => x.status === "ok").length;
      setNotice({ type: "ok", text: `Đã tiếp nhận ${res.uploaded.length} tệp (${okCount} hồ sơ trích xuất văn bản thành công).` });
    } catch (err) {
      setNotice({ type: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  }

  async function handleEvaluateAll() {
    if (!activeJob) return;
    setBusy(true);
    setNotice(null);
    try {
      const res = await evaluateJob(activeJob.id);
      setEvalTimings(res.timings || null);
      const rk = await fetchRanking(activeJob.id, levelFilter);
      setRanking(rk.items || []);
      setStep(3);
      setNotice({
        type: "ok",
        text: `Đã chấm điểm ${res.evaluated_count} hồ sơ trong ${res.timings?.total_ms || 0} ms (${res.timings?.per_cv_ms || 0} ms/hồ sơ).`,
      });
    } catch (err) {
      setNotice({ type: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  }

  async function handleFilterChange(lvl) {
    setLevelFilter(lvl);
    if (!activeJob) return;
    const rk = await fetchRanking(activeJob.id, lvl);
    setRanking(rk.items || []);
  }

  async function handleOpenDetail(evalId) {
    setBusy(true);
    try {
      const data = await fetchEvaluationDetail(evalId);
      setDetail(data);
      setStep(4);
    } catch (err) {
      setNotice({ type: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteCandidate(candidateId) {
    setBusy(true);
    try {
      await deleteCandidate(candidateId);
      if (activeJob) {
        setCvs(await fetchJobCvs(activeJob.id));
        const rk = await fetchRanking(activeJob.id, levelFilter);
        setRanking(rk.items || []);
      }
      if (detail && detail.candidate_id === candidateId) {
        setDetail(null);
        setStep(3);
      }
      setNotice({ type: "ok", text: "Đã xóa hồ sơ ứng viên và kết quả đánh giá khỏi hệ thống." });
    } catch (err) {
      setNotice({ type: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  }

  const levelCounts = ranking.reduce(
    (acc, item) => {
      acc[item.level] = (acc[item.level] || 0) + 1;
      return acc;
    },
    { 0: 0, 1: 0, 2: 0 }
  );

  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="rail-brand">
          <h1>Sàng lọc Hồ sơ</h1>
          <p>Đánh giá phù hợp CV theo tiêu chí</p>
        </div>
        <button className="rail-new-btn" onClick={startNewJob}>
          + Tạo vị trí mới
        </button>
        <div className="rail-section-title">Vị trí đã lưu ({jobs.length})</div>
        <div className="rail-job-list">
          {jobs.map((j) => (
            <button
              key={j.id}
              className={`rail-job-item ${activeJob?.id === j.id ? "active" : ""}`}
              onClick={() => {
                openJob(j.id);
                setStep(1);
              }}
            >
              <span className="rail-job-title">{j.title}</span>
              <span className="rail-job-meta num">{j.candidate_count} hồ sơ</span>
            </button>
          ))}
        </div>
        {health && (
          <div className="rail-model-box">
            <strong>{health.model_version}</strong>
            <div className="num">Ngưỡng: t1 = {health.thresholds.t1}, t2 = {health.thresholds.t2}</div>
            <div className="num">Luật bắt buộc: ρ = {health.must_ratio}, τ = {health.skill_threshold}</div>
          </div>
        )}
      </aside>

      <main className="workspace">
        <nav className="step-bar" aria-label="Quy trình sàng lọc">
          <button className={`step-tab ${step === 1 ? "active" : ""}`} onClick={() => setStep(1)}>
            <span className="step-num">1</span> Yêu cầu vị trí
          </button>
          <button className={`step-tab ${step === 2 ? "active" : ""}`} disabled={!activeJob} onClick={() => setStep(2)}>
            <span className="step-num">2</span> Tải hồ sơ ({cvs.length})
          </button>
          <button className={`step-tab ${step === 3 ? "active" : ""}`} disabled={!activeJob} onClick={() => setStep(3)}>
            <span className="step-num">3</span> Bảng xếp hạng ({ranking.length})
          </button>
          <button className={`step-tab ${step === 4 ? "active" : ""}`} disabled={!detail} onClick={() => setStep(4)}>
            <span className="step-num">4</span> Chi tiết ứng viên
          </button>
        </nav>

        {notice && (
          <div className={`notice-bar ${notice.type === "error" ? "notice-error" : "notice-ok"}`} role="status">
            {notice.text}
          </div>
        )}

        {step === 1 && (
          <section>
            <div className="page-header">
              <div>
                <h2>{activeJob ? activeJob.title : "Tạo vị trí tuyển dụng mới"}</h2>
                <p>Nhập mô tả công việc để trích xuất yêu cầu, sau đó điều chỉnh danh sách kỹ năng bắt buộc trước khi chấm hồ sơ.</p>
              </div>
              {activeJob && (
                <button className="btn-primary" onClick={() => setStep(2)}>
                  Chuyển sang tải hồ sơ
                </button>
              )}
            </div>

            <div className="two-col">
              <form className="panel" onSubmit={handleCreateJob}>
                <h3 className="panel-title">Mô tả công việc (JD)</h3>
                <div className="field">
                  <label htmlFor="jd-title">Tên vị trí</label>
                  <input
                    id="jd-title"
                    className="input"
                    value={jdTitle}
                    onChange={(e) => setJdTitle(e.target.value)}
                    placeholder="Ví dụ: Chuyên viên Phân tích Dữ liệu"
                  />
                </div>
                <div className="field">
                  <label htmlFor="jd-desc">Nội dung yêu cầu chi tiết</label>
                  <textarea
                    id="jd-desc"
                    className="textarea"
                    value={jdDesc}
                    onChange={(e) => setJdDesc(e.target.value)}
                    placeholder="Dán mô tả nhiệm vụ, kỹ năng bắt buộc, ưu tiên, số năm kinh nghiệm và bằng cấp..."
                  />
                </div>
                <div className="btn-row">
                  <button type="submit" className="btn-primary" disabled={busy}>
                    {busy ? "Đang phân tích..." : "Phân tích mô tả công việc"}
                  </button>
                  <button
                    type="button"
                    className="btn-secondary"
                    onClick={() => {
                      setJdTitle(SAMPLE_JD.title);
                      setJdDesc(SAMPLE_JD.description);
                    }}
                  >
                    Điền mẫu Data Analyst
                  </button>
                </div>
              </form>

              <div className="panel">
                <h3 className="panel-title">Yêu cầu đã trích xuất</h3>
                {!activeJob ? (
                  <p style={{ color: "var(--ink-muted)" }}>
                    Phân tích mô tả công việc ở cột bên trái để xem và chỉnh sửa danh sách yêu cầu bắt buộc.
                  </p>
                ) : (
                  <>
                    <div className="two-col" style={{ gap: 12, marginBottom: 14 }}>
                      <div className="field" style={{ marginBottom: 0 }}>
                        <label htmlFor="req-years">Số năm kinh nghiệm tối thiểu</label>
                        <input
                          id="req-years"
                          type="number"
                          step="0.5"
                          min="0"
                          className="input num"
                          value={reqYears}
                          onChange={(e) => setReqYears(e.target.value)}
                        />
                      </div>
                      <div className="field" style={{ marginBottom: 0 }}>
                        <label htmlFor="req-deg">Bậc học tối thiểu</label>
                        <select id="req-deg" className="select" value={reqDegree} onChange={(e) => setReqDegree(e.target.value)}>
                          <option value="none">Không yêu cầu</option>
                          <option value="high_school">Trung học phổ thông</option>
                          <option value="associate">Cao đẳng</option>
                          <option value="bachelor">Đại học / Cử nhân</option>
                          <option value="master">Thạc sĩ</option>
                          <option value="phd">Tiến sĩ</option>
                        </select>
                      </div>
                    </div>

                    <div className="btn-row" style={{ marginBottom: 12 }}>
                      <input
                        className="input"
                        style={{ flex: 1 }}
                        value={newSkillText}
                        onChange={(e) => setNewSkillText(e.target.value)}
                        placeholder="Thêm kỹ năng yêu cầu (ví dụ: postgresql)"
                      />
                      <button type="button" className="btn-secondary" onClick={handleAddSkillReq}>
                        Thêm kỹ năng
                      </button>
                    </div>

                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Yêu cầu</th>
                          <th>Loại</th>
                          <th>Bắt buộc</th>
                        </tr>
                      </thead>
                      <tbody>
                        {reqs.map((r, idx) => (
                          <tr key={r.id || idx}>
                            <td>{r.text}</td>
                            <td>
                              <span className="badge badge-neutral">{r.type}</span>
                            </td>
                            <td>
                              <input
                                type="checkbox"
                                checked={Boolean(r.is_mandatory)}
                                onChange={(e) => {
                                  const next = [...reqs];
                                  next[idx] = { ...next[idx], is_mandatory: e.target.checked };
                                  setReqs(next);
                                }}
                              />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>

                    <div className="btn-row" style={{ marginTop: 14 }}>
                      <button type="button" className="btn-primary" disabled={busy} onClick={handleSaveRequirements}>
                        Lưu yêu cầu bắt buộc
                      </button>
                    </div>
                  </>
                )}
              </div>
            </div>
          </section>
        )}

        {step === 2 && activeJob && (
          <section>
            <div className="page-header">
              <div>
                <h2>Tải lên và trích xuất hồ sơ ứng viên</h2>
                <p>Hỗ trợ nhiều tệp PDF có lớp chữ hoặc DOCX cùng lúc (tối đa 5 MB mỗi tệp). Tệp trùng nội dung được phát hiện tự động.</p>
              </div>
              <button className="btn-primary" disabled={busy || cvs.filter((c) => c.status === "ok").length === 0} onClick={handleEvaluateAll}>
                {busy ? "Đang chấm điểm..." : "Chấm điểm hồ sơ"}
              </button>
            </div>

            <div className="panel" style={{ marginBottom: 18 }}>
              <div className="dropzone">
                <p>Chọn một hoặc nhiều tệp CV định dạng PDF hoặc DOCX để tải vào vị trí <strong>{activeJob.title}</strong>.</p>
                <input
                  id="cv-file-input"
                  type="file"
                  accept=".pdf,.docx"
                  multiple
                  onChange={(e) => handleFileSelect(e.target.files)}
                />
              </div>

              {uploadReport.length > 0 && (
                <div className="notice-bar notice-warn">
                  Kết quả tải gần nhất:{" "}
                  {uploadReport.map((u) => `${u.file_name} (${u.status})`).join("; ")}
                </div>
              )}

              <table className="data-table">
                <thead>
                  <tr>
                    <th>Mã</th>
                    <th>Tên tệp</th>
                    <th>Thông tin nhận diện (đã ẩn liên hệ)</th>
                    <th>Trạng thái</th>
                    <th>Ghi chú</th>
                  </tr>
                </thead>
                <tbody>
                  {cvs.length === 0 ? (
                    <tr>
                      <td colSpan={5} style={{ color: "var(--ink-muted)" }}>
                        Chưa có hồ sơ nào được tải lên cho vị trí này.
                      </td>
                    </tr>
                  ) : (
                    cvs.map((c) => (
                      <tr key={c.id}>
                        <td className="num">#{c.id}</td>
                        <td>{c.file_name}</td>
                        <td>{c.masked_contact}</td>
                        <td>
                          <span className={`badge ${c.status === "ok" ? "badge-level-2" : "badge-level-0"}`}>
                            {c.status === "ok" ? "Hợp lệ" : "Không đọc được"}
                          </span>
                        </td>
                        <td style={{ color: "var(--ink-muted)" }}>{c.error_message || "Sẵn sàng chấm điểm"}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {step === 3 && activeJob && (
          <section>
            <div className="page-header">
              <div>
                <h2>Bảng xếp hạng ứng viên</h2>
                <p>Xếp hạng giảm dần theo điểm cuối cùng (phá hòa theo điểm tổng hợp thô). Vạch đứng trên dải điểm biểu thị ngưỡng Phù hợp t2 = {health?.thresholds.t2 || 67}.</p>
              </div>
              <button className="btn-primary" disabled={busy} onClick={handleEvaluateAll}>
                Chấm lại toàn bộ
              </button>
            </div>

            <div className="summary-strip">
              <div className="summary-cell">
                <div className="label">Hồ sơ đã chấm</div>
                <div className="value num">{ranking.length}</div>
              </div>
              <div className="summary-cell">
                <div className="label">Phù hợp (Good Fit)</div>
                <div className="value num" style={{ color: "var(--fit-good-fg)" }}>{levelCounts[2] || 0}</div>
              </div>
              <div className="summary-cell">
                <div className="label">Tiềm năng (Potential)</div>
                <div className="value num" style={{ color: "var(--fit-potential-fg)" }}>{levelCounts[1] || 0}</div>
              </div>
              <div className="summary-cell">
                <div className="label">Không phù hợp (No Fit)</div>
                <div className="value num" style={{ color: "var(--fit-none-fg)" }}>{levelCounts[0] || 0}</div>
              </div>
            </div>

            <div className="filter-pills">
              {[
                ["all", "Tất cả"],
                ["2", "Phù hợp"],
                ["1", "Tiềm năng"],
                ["0", "Không phù hợp"],
              ].map(([val, label]) => (
                <button
                  key={val}
                  className={`filter-pill ${levelFilter === val ? "active" : ""}`}
                  onClick={() => handleFilterChange(val)}
                >
                  {label}
                </button>
              ))}
            </div>

            <div className="panel">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Hạng</th>
                    <th>Hồ sơ (đã ẩn liên hệ)</th>
                    <th>Điểm tổng</th>
                    <th>Mức phù hợp</th>
                    <th>Ngữ nghĩa / Kỹ năng / KN</th>
                    <th>Điều kiện bắt buộc</th>
                    <th>Thao tác</th>
                  </tr>
                </thead>
                <tbody>
                  {ranking.map((item) => {
                    const capped = item.must_have_failed && item.must_have_failed.length > 0 && item.raw_score > item.final_score;
                    return (
                      <tr key={item.eval_id}>
                        <td className="num"><strong>#{item.rank}</strong></td>
                        <td>
                          <div><strong>{item.file_name}</strong></div>
                          <div style={{ fontSize: 12.5, color: "var(--ink-muted)" }}>{item.masked_contact}</div>
                        </td>
                        <td className="num">
                          <strong>{fmtScore(item.final_score)}</strong>
                          {capped && (
                            <div style={{ fontSize: 12, color: "var(--fit-none-fg)" }}>
                              Gốc: {fmtScore(item.raw_score)} (bị chặn)
                            </div>
                          )}
                        </td>
                        <td>
                          <span className={`badge badge-level-${item.level}`}>{item.level_label}</span>
                        </td>
                        <td className="num">
                          {fmtScore(item.criteria.sem)} / {fmtScore(item.criteria.skill)} / {fmtScore(item.criteria.exp)}
                        </td>
                        <td>
                          {item.must_have_failed.length === 0 ? (
                            <span style={{ color: "var(--fit-good-fg)", fontSize: 13 }}>Đạt yêu cầu</span>
                          ) : (
                            <span style={{ color: "var(--fit-none-fg)", fontSize: 13 }}>
                              Vi phạm ({item.must_have_failed.join(", ")})
                            </span>
                          )}
                        </td>
                        <td>
                          <div className="btn-row">
                            <button className="btn-secondary" onClick={() => handleOpenDetail(item.eval_id)}>
                              Xem giải thích
                            </button>
                            <button className="btn-danger-link" onClick={() => handleDeleteCandidate(item.candidate_id)}>
                              Xóa
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {step === 4 && detail && (
          <section>
            <div className="page-header">
              <div>
                <h2>Chi tiết đánh giá: {detail.file_name}</h2>
                <p>{detail.masked_contact}</p>
              </div>
              <button className="btn-secondary" onClick={() => setStep(3)}>
                Quay lại bảng xếp hạng
              </button>
            </div>

            {detail.must_have_failed && detail.must_have_failed.length > 0 && (
              <div className="notice-bar notice-error">
                Luật điều kiện bắt buộc đã kích hoạt ({detail.must_have_failed.join(", ")}): điểm bị giới hạn từ{" "}
                <strong className="num">{fmtScore(detail.raw_score)}</strong> xuống{" "}
                <strong className="num">{fmtScore(detail.final_score)}</strong> (dưới ngưỡng Phù hợp{" "}
                <span className="num">{detail.thresholds.t2}</span>).
              </div>
            )}

            <div className="score-hero-grid">
              <div className="panel score-box">
                <div>
                  <div style={{ color: "var(--ink-muted)", fontSize: 13 }}>Điểm đánh giá cuối cùng</div>
                  <div className="big-score num">{fmtScore(detail.final_score)}</div>
                  <span className={`badge badge-level-${detail.level}`}>{detail.level_label}</span>
                </div>

                <div>
                  <div className="threshold-track">
                    <div
                      className="threshold-marker"
                      style={{ left: `${Math.min(100, Math.max(0, detail.final_score))}%` }}
                      title={`Điểm cuối: ${detail.final_score}`}
                    />
                    {detail.raw_score > detail.final_score && (
                      <div
                        className="threshold-raw-marker"
                        style={{ left: `${Math.min(100, Math.max(0, detail.raw_score))}%` }}
                        title={`Điểm trước khi chặn: ${detail.raw_score}`}
                      />
                    )}
                  </div>
                  <div className="num" style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--ink-muted)" }}>
                    <span>0</span>
                    <span>t1 = {detail.thresholds.t1}</span>
                    <span>t2 = {detail.thresholds.t2}</span>
                    <span>100</span>
                  </div>
                </div>

                <div className="num" style={{ fontSize: 12.5, color: "var(--ink-muted)", marginTop: 10 }}>
                  Mô hình: {detail.model_version} | Thời gian: {detail.elapsed_ms} ms
                </div>
              </div>

              <div className="panel">
                <h3 className="panel-title">Điểm năm tiêu chí và trọng số (vạch xám = ngưỡng t2)</h3>
                {CRITERIA_LABELS.map(([key, label]) => {
                  const val = detail.criteria[key];
                  const w = detail.weights?.[key] ?? 0;
                  return (
                    <div className="criteria-row" key={key}>
                      <span>
                        {label} <span className="num" style={{ color: "var(--ink-muted)", fontSize: 12 }}>({Math.round(w * 100)}%)</span>
                      </span>
                      <div className="criteria-track">
                        <div className="criteria-fill" style={{ width: `${val === null || val === undefined ? 0 : Math.min(100, val)}%` }} />
                        <div className="criteria-t2-line" style={{ left: `${detail.thresholds.t2}%` }} />
                      </div>
                      <span className="num" style={{ textAlign: "right", fontWeight: 600 }}>
                        {val === null || val === undefined ? "—" : fmtScore(val)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="two-col" style={{ marginBottom: 20 }}>
              <div className="panel">
                <h3 className="panel-title">Kỹ năng đáp ứng ({detail.matched.length})</h3>
                <div className="skill-chip-wrap">
                  {detail.matched.length === 0 ? (
                    <span style={{ color: "var(--ink-muted)" }}>Không có kỹ năng trùng khớp.</span>
                  ) : (
                    detail.matched.map((m, i) => (
                      <span key={i} className="skill-chip skill-matched num">
                        {m.skill} ({m.type === "must" ? "bắt buộc" : "ưu tiên"}: {fmtScore(m.score * 100)})
                      </span>
                    ))
                  )}
                </div>
              </div>

              <div className="panel">
                <h3 className="panel-title">Kỹ năng còn thiếu ({detail.missing.length})</h3>
                <div className="skill-chip-wrap">
                  {detail.missing.length === 0 ? (
                    <span style={{ color: "var(--fit-good-fg)" }}>Đáp ứng đầy đủ kỹ năng yêu cầu.</span>
                  ) : (
                    detail.missing.map((m, i) => (
                      <span key={i} className="skill-chip skill-missing">
                        {m.skill} ({m.type === "must" ? "bắt buộc" : "ưu tiên"})
                      </span>
                    ))
                  )}
                </div>
              </div>
            </div>

            <div className="panel">
              <h3 className="panel-title">Đoạn CV khớp nhất với từng đoạn yêu cầu của JD</h3>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Đoạn JD</th>
                    <th>Đoạn CV đáp ứng cao nhất</th>
                    <th>Độ tương đồng</th>
                  </tr>
                </thead>
                <tbody>
                  {(detail.best_matches || []).map((bm) => (
                    <tr key={bm.jd_chunk_idx}>
                      <td><div className="excerpt-box">{bm.jd_excerpt}</div></td>
                      <td><div className="excerpt-box">{bm.cv_excerpt}</div></td>
                      <td className="num"><strong>{bm.similarity.toFixed(3).replace(".", ",")}</strong></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
