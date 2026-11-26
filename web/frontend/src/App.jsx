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

  // busy: true = thao tác nhanh; chuỗi = thao tác chậm, hiện hộp chờ với nội dung đó.
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

  async function handleCreateJob(e, file = null) {
    e.preventDefault();
    setBusy(file ? `Đang đọc tệp JD ${file.name}` : "Đang phân tích mô tả công việc");
    setNotice(null);
    try {
      // Tải tệp khi đang mở một vị trí cũ: bỏ tên cũ để lấy tên từ chính tệp JD.
      const title = file && activeJob ? "" : jdTitle;
      const created = await createJob({ title, description: jdDesc, file });
      await refreshJobs(created.id);
      setNotice({ type: "ok", text: `Đã phân tích và tạo vị trí "${created.title}". Kiểm tra lại các yêu cầu bắt buộc ở cột bên phải trước khi tải hồ sơ.` });
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
      setNotice({ type: "ok", text: "Đã lưu yêu cầu. Lần chấm tiếp theo sẽ dùng danh sách mới." });
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
    setBusy(fileList.length === 1 ? `Đang đọc hồ sơ ${fileList[0].name}` : `Đang đọc ${fileList.length} hồ sơ`);
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
    setBusy(`Đang chấm điểm ${cvs.filter((c) => c.status === "ok").length} hồ sơ`);
    setNotice(null);
    try {
      const res = await evaluateJob(activeJob.id);
      setEvalTimings(res.timings || null);
      const rk = await fetchRanking(activeJob.id);
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

  // Lọc phía client để thước điểm luôn thấy đủ mọi ứng viên và giữ hạng gốc.
  function handleFilterChange(lvl) {
    setLevelFilter(lvl);
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
        const rk = await fetchRanking(activeJob.id);
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
  const visibleRanking = levelFilter === "all" ? ranking : ranking.filter((r) => String(r.level) === levelFilter);
  const t1 = health?.thresholds.t1 ?? 33;
  const t2 = health?.thresholds.t2 ?? 67;
  const readyCvCount = cvs.filter((c) => c.status === "ok").length;

  const steps = [
    { n: 1, label: "Yêu cầu vị trí", enabled: true },
    { n: 2, label: "Hồ sơ", count: cvs.length, enabled: Boolean(activeJob) },
    { n: 3, label: "Xếp hạng", count: ranking.length, enabled: Boolean(activeJob) },
    { n: 4, label: "Chi tiết ứng viên", enabled: Boolean(detail) },
  ];

  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="brand">
          <span className="brand-name">Sàng lọc hồ sơ</span>
          <span className="brand-sub">So khớp CV với mô tả công việc</span>
        </div>

        <button className="btn-new" onClick={startNewJob}>
          Tạo vị trí mới
        </button>

        <h2 className="rail-heading">Vị trí đã lưu</h2>
        {jobs.length === 0 ? (
          <p className="rail-empty">Chưa có vị trí nào. Tạo vị trí đầu tiên để bắt đầu.</p>
        ) : (
          <ul className="job-index">
            {jobs.map((j) => (
              <li key={j.id}>
                <button
                  className={`job-entry ${activeJob?.id === j.id ? "is-active" : ""}`}
                  aria-current={activeJob?.id === j.id ? "true" : undefined}
                  onClick={() => {
                    openJob(j.id);
                    setStep(1);
                  }}
                >
                  <span className="job-entry-title">{j.title}</span>
                  <span className="job-entry-count num">{j.candidate_count}</span>
                </button>
              </li>
            ))}
          </ul>
        )}

        {health && (
          <dl className="model-slip num">
            <dt>Mô hình</dt>
            <dd>{health.model_version}</dd>
            <dt>Ngưỡng</dt>
            <dd>t1 {health.thresholds.t1}, t2 {health.thresholds.t2}</dd>
            <dt>Luật bắt buộc</dt>
            <dd>ρ {health.must_ratio}, τ {health.skill_threshold}</dd>
          </dl>
        )}
      </aside>

      <main className="workspace">
        <nav className="tabs" aria-label="Các bước sàng lọc">
          {steps.map((s) => (
            <button
              key={s.n}
              className={`tab ${step === s.n ? "is-active" : ""}`}
              aria-current={step === s.n ? "step" : undefined}
              disabled={!s.enabled}
              onClick={() => setStep(s.n)}
            >
              <span className="tab-n num">{s.n}</span>
              {s.label}
              {s.count !== undefined && <span className="tab-count num">{s.count}</span>}
            </button>
          ))}
        </nav>

        <div className="sheet">
          {notice && (
            <div className={`notice ${notice.type === "error" ? "notice-error" : "notice-ok"}`} role="status">
              {notice.text}
              <button className="notice-close" aria-label="Đóng thông báo" onClick={() => setNotice(null)}>×</button>
            </div>
          )}

          {step === 1 && (
            <section>
              <header className="sheet-head">
                <div>
                  <h1>{activeJob ? activeJob.title : "Vị trí tuyển dụng mới"}</h1>
                  <p className="lede">
                    Tải tệp JD hoặc dán nội dung, hệ thống sẽ tách ra kỹ năng, số năm kinh nghiệm và bằng cấp. Bạn sửa lại danh sách trước khi chấm hồ sơ.
                  </p>
                </div>
                {activeJob && (
                  <button className="btn-primary" onClick={() => setStep(2)}>
                    Tải hồ sơ
                  </button>
                )}
              </header>

              <div className="split">
                <form className="block" onSubmit={handleCreateJob}>
                  <h2 className="block-title">Mô tả công việc</h2>
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
                  <label
                    className="drop drop-small"
                    htmlFor="jd-file-input"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => e.dataTransfer.files[0] && handleCreateJob(e, e.dataTransfer.files[0])}
                  >
                    <span className="drop-main">{busy ? "Đang đọc tệp JD..." : "Tải tệp JD"}</span>
                    <span className="drop-sub">PDF hoặc DOCX, tối đa 5 MB. Hoặc dán nội dung vào ô bên dưới.</span>
                    <input
                      id="jd-file-input"
                      type="file"
                      accept=".pdf,.docx"
                      disabled={busy}
                      onChange={(e) => {
                        const f = e.target.files[0];
                        e.target.value = ""; // cho phép chọn lại cùng tệp
                        if (f) handleCreateJob(e, f);
                      }}
                    />
                  </label>
                  <div className="field">
                    <label htmlFor="jd-desc">Nội dung</label>
                    <textarea
                      id="jd-desc"
                      className="textarea"
                      value={jdDesc}
                      onChange={(e) => setJdDesc(e.target.value)}
                      placeholder="Nhiệm vụ, kỹ năng bắt buộc, kỹ năng ưu tiên, số năm kinh nghiệm, bằng cấp..."
                    />
                  </div>
                  <div className="actions">
                    <button type="submit" className="btn-primary" disabled={busy}>
                      {busy ? "Đang phân tích..." : "Phân tích mô tả"}
                    </button>
                    <button
                      type="button"
                      className="btn-quiet"
                      onClick={() => {
                        setJdTitle(SAMPLE_JD.title);
                        setJdDesc(SAMPLE_JD.description);
                      }}
                    >
                      Dùng mẫu Data Analyst
                    </button>
                  </div>
                </form>

                <div className="block">
                  <h2 className="block-title">Yêu cầu đã tách</h2>
                  {!activeJob ? (
                    <p className="muted">Sau khi phân tích, danh sách yêu cầu sẽ hiện ở đây để bạn chỉnh.</p>
                  ) : (
                    <>
                      <div className="field-pair">
                        <div className="field">
                          <label htmlFor="req-years">Kinh nghiệm tối thiểu (năm)</label>
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
                        <div className="field">
                          <label htmlFor="req-deg">Bậc học tối thiểu</label>
                          <select id="req-deg" className="input" value={reqDegree} onChange={(e) => setReqDegree(e.target.value)}>
                            <option value="none">Không yêu cầu</option>
                            <option value="high_school">Trung học phổ thông</option>
                            <option value="associate">Cao đẳng</option>
                            <option value="bachelor">Đại học / Cử nhân</option>
                            <option value="master">Thạc sĩ</option>
                            <option value="phd">Tiến sĩ</option>
                          </select>
                        </div>
                      </div>

                      <ul className="req-list">
                        {reqs.map((r, idx) => (
                          <li key={r.id || idx} className={r.is_mandatory ? "is-must" : ""}>
                            <label>
                              <input
                                type="checkbox"
                                checked={Boolean(r.is_mandatory)}
                                onChange={(e) => {
                                  const next = [...reqs];
                                  next[idx] = { ...next[idx], is_mandatory: e.target.checked };
                                  setReqs(next);
                                }}
                              />
                              <span className="req-text">{r.text}</span>
                            </label>
                            <span className="req-type">{TYPE_LABELS[r.type] || r.type}</span>
                          </li>
                        ))}
                      </ul>
                      <p className="hint">Đánh dấu = bắt buộc. Thiếu yêu cầu bắt buộc thì điểm bị giới hạn dưới ngưỡng Phù hợp.</p>

                      <form
                        className="add-skill"
                        onSubmit={(e) => {
                          e.preventDefault();
                          handleAddSkillReq();
                        }}
                      >
                        <label htmlFor="new-skill" className="visually-hidden">Thêm kỹ năng</label>
                        <input
                          id="new-skill"
                          className="input"
                          value={newSkillText}
                          onChange={(e) => setNewSkillText(e.target.value)}
                          placeholder="Thêm kỹ năng, ví dụ postgresql"
                        />
                        <button type="submit" className="btn-quiet">Thêm</button>
                      </form>

                      <div className="actions">
                        <button type="button" className="btn-primary" disabled={busy} onClick={handleSaveRequirements}>
                          Lưu yêu cầu
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
              <header className="sheet-head">
                <div>
                  <h1>Hồ sơ cho {activeJob.title}</h1>
                  <p className="lede">PDF có lớp chữ hoặc DOCX, tối đa 5 MB mỗi tệp. Tệp trùng nội dung sẽ bị bỏ qua. Email và số điện thoại được ẩn trước khi chấm.</p>
                </div>
                <button className="btn-primary" disabled={busy || readyCvCount === 0} onClick={handleEvaluateAll}>
                  {busy ? "Đang chấm..." : `Chấm ${readyCvCount} hồ sơ`}
                </button>
              </header>

              <label
                className="drop"
                htmlFor="cv-file-input"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  handleFileSelect(e.dataTransfer.files);
                }}
              >
                <span className="drop-main">Chọn tệp CV</span>
                <span className="drop-sub">hoặc kéo nhiều tệp vào đây</span>
                <input
                  id="cv-file-input"
                  type="file"
                  accept=".pdf,.docx"
                  multiple
                  onChange={(e) => handleFileSelect(e.target.files)}
                />
              </label>

              {uploadReport.length > 0 && (
                <p className="upload-report">
                  Lần tải gần nhất: {uploadReport.map((u) => `${u.file_name} (${u.status})`).join("; ")}
                </p>
              )}

              <div className="table-wrap">
                <table className="ledger">
                  <thead>
                    <tr>
                      <th className="num">Mã</th>
                      <th>Tệp</th>
                      <th>Liên hệ (đã ẩn)</th>
                      <th>Trạng thái</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cvs.length === 0 ? (
                      <tr>
                        <td colSpan={4} className="muted">Chưa có hồ sơ nào. Chọn tệp ở trên để tải lên.</td>
                      </tr>
                    ) : (
                      cvs.map((c) => (
                        <tr key={c.id}>
                          <td className="num muted">{c.id}</td>
                          <td className="strong">{c.file_name}</td>
                          <td className="muted">{c.masked_contact}</td>
                          <td>
                            {c.status === "ok" ? (
                              <span className="status-ok">Sẵn sàng chấm</span>
                            ) : (
                              <span className="pen">{c.error_message || "Không đọc được nội dung"}</span>
                            )}
                          </td>
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
              <header className="sheet-head">
                <div>
                  <h1>Xếp hạng ứng viên</h1>
                  <p className="lede">
                    Sắp theo điểm cuối, hòa thì xét điểm trước khi giới hạn. Bấm vào một khoảng trên thước để lọc theo mức phù hợp.
                  </p>
                </div>
                <button className="btn-quiet" disabled={busy} onClick={handleEvaluateAll}>
                  {busy ? "Đang chấm..." : "Chấm lại"}
                </button>
              </header>

              {ranking.length === 0 ? (
                <p className="empty">Chưa có kết quả. Tải hồ sơ ở bước 2 rồi bấm Chấm.</p>
              ) : (
                <>
                  <ScoreRuler
                    items={ranking}
                    t1={t1}
                    t2={t2}
                    counts={levelCounts}
                    filter={levelFilter}
                    onFilter={handleFilterChange}
                  />

                  <div className="table-wrap">
                    <table className="ledger ranking">
                      <thead>
                        <tr>
                          <th className="num">Hạng</th>
                          <th>Ứng viên</th>
                          <th className="num">Điểm</th>
                          <th>Mức</th>
                          <th className="num" title="Ngữ nghĩa, Kỹ năng, Kinh nghiệm">NN · KN · KNg</th>
                          <th>Yêu cầu bắt buộc</th>
                          <th><span className="visually-hidden">Thao tác</span></th>
                        </tr>
                      </thead>
                      <tbody>
                        {visibleRanking.map((item) => {
                          const capped = item.must_have_failed?.length > 0 && item.raw_score > item.final_score;
                          return (
                            <tr key={item.eval_id}>
                              <td className="num rank">{item.rank}</td>
                              <td>
                                <div className="strong">{item.file_name}</div>
                                <div className="muted small">{item.masked_contact}</div>
                              </td>
                              <td className="num score-cell">
                                {capped && <s className="raw-struck">{fmtScore(item.raw_score)}</s>}
                                <span className={capped ? "pen-score" : "final"}>{fmtScore(item.final_score)}</span>
                              </td>
                              <td>
                                <span className={`level level-${item.level}`}>{item.level_label}</span>
                              </td>
                              <td className="num muted">
                                {fmtShort(item.criteria.sem)} · {fmtShort(item.criteria.skill)} · {fmtShort(item.criteria.exp)}
                              </td>
                              <td>
                                {item.must_have_failed.length === 0 ? (
                                  <span className="status-ok">Đủ</span>
                                ) : (
                                  <span className="pen">Thiếu {item.must_have_failed.join(", ")}</span>
                                )}
                              </td>
                              <td className="row-actions">
                                <button className="btn-link" onClick={() => handleOpenDetail(item.eval_id)}>
                                  Xem lý do
                                </button>
                                <button className="btn-link btn-link-danger" onClick={() => handleDeleteCandidate(item.candidate_id)}>
                                  Xóa
                                </button>
                              </td>
                            </tr>
                          );
                        })}
                        {visibleRanking.length === 0 && (
                          <tr>
                            <td colSpan={7} className="muted">Không có ứng viên ở mức này.</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </section>
          )}

          {step === 4 && detail && <Detail detail={detail} onBack={() => setStep(3)} />}
        </div>
      </main>

      {typeof busy === "string" && (
        <div className="overlay">
          <div className="overlay-box" role="status" aria-live="polite">
            <span className="spinner" aria-hidden="true" />
            <p className="overlay-title">{busy}</p>
            <p className="overlay-sub">
              Mỗi tệp mất vài giây, trang vẫn đang chạy. Vui lòng không đóng hoặc tải lại trang.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

const TYPE_LABELS = {
  skill: "kỹ năng",
  experience: "kinh nghiệm",
  education: "học vấn",
  cert: "chứng chỉ",
};

function fmtShort(val) {
  if (val === null || val === undefined) return "–";
  return Math.round(Number(val));
}

const clampPct = (v) => Math.min(100, Math.max(0, Number(v) || 0));

function ScoreRuler({ items, t1, t2, counts, filter, onFilter }) {
  const bands = [
    { level: "0", from: 0, to: t1, label: "Không phù hợp" },
    { level: "1", from: t1, to: t2, label: "Tiềm năng" },
    { level: "2", from: t2, to: 100, label: "Phù hợp" },
  ];
  return (
    <figure className="ruler" aria-label="Phân bố điểm ứng viên trên thang 0 đến 100">
      <div className="ruler-bands">
        {bands.map((b) => (
          <button
            key={b.level}
            className={`band band-${b.level} ${filter === b.level ? "is-on" : ""}`}
            style={{ left: `${b.from}%`, width: `${b.to - b.from}%` }}
            aria-pressed={filter === b.level}
            onClick={() => onFilter(filter === b.level ? "all" : b.level)}
          >
            <span className="band-count num">{counts[b.level] || 0}</span>
            <span className="band-label">{b.label}</span>
          </button>
        ))}
      </div>
      <div className="ruler-scale" aria-hidden="true">
        {items.map((it, i) => {
          const capped = it.raw_score > it.final_score && it.must_have_failed?.length > 0;
          const dim = filter !== "all" && String(it.level) !== filter;
          return (
            <span key={it.eval_id} className={dim ? "is-dim" : ""}>
              {capped && (
                <span
                  className="cap-line"
                  style={{ left: `${clampPct(it.final_score)}%`, width: `${clampPct(it.raw_score) - clampPct(it.final_score)}%` }}
                />
              )}
              <span
                className={`tick tick-${it.level}`}
                style={{ left: `${clampPct(it.final_score)}%`, animationDelay: `${i * 35}ms` }}
                title={`#${it.rank} ${it.file_name}: ${fmtScore(it.final_score)}`}
              />
            </span>
          );
        })}
        {[t1, t2].map((t) => (
          <span key={t} className="threshold" style={{ left: `${t}%` }}>
            <span className="num">{t}</span>
          </span>
        ))}
      </div>
      <div className="ruler-axis num" aria-hidden="true">
        {[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
          .filter((v) => Math.abs(v - t1) > 4 && Math.abs(v - t2) > 4)
          .map((v) => (
            <span key={v} className={v % 20 ? "minor" : ""} style={{ left: `${v}%` }}>{v}</span>
          ))}
      </div>
      <figcaption className="muted small">
        Mỗi vạch là một ứng viên. Đường đỏ nối điểm gốc với điểm sau khi bị giới hạn do thiếu yêu cầu bắt buộc.
        {filter !== "all" && (
          <>
            {" "}
            <button className="btn-link" onClick={() => onFilter("all")}>Bỏ lọc</button>
          </>
        )}
      </figcaption>
    </figure>
  );
}

function Detail({ detail, onBack }) {
  const capped = detail.must_have_failed?.length > 0 && detail.raw_score > detail.final_score;
  const { t1, t2 } = detail.thresholds;
  return (
    <section>
      <header className="sheet-head">
        <div>
          <h1>{detail.file_name}</h1>
          <p className="lede">{detail.masked_contact}</p>
        </div>
        <button className="btn-quiet" onClick={onBack}>
          Về bảng xếp hạng
        </button>
      </header>

      <div className="verdict">
        <div className="verdict-score">
          <span className={`level level-${detail.level}`}>{detail.level_label}</span>
          <div className="big-score num">
            {fmtScore(detail.final_score)}
            {capped && <s className="big-raw">{fmtScore(detail.raw_score)}</s>}
          </div>
          {capped && (
            <p className="pen pen-note">
              Thiếu {detail.must_have_failed.join(", ")}, nên điểm bị hạ từ {fmtScore(detail.raw_score)} xuống dưới ngưỡng Phù hợp ({t2}).
            </p>
          )}
          <div className="mini-ruler" aria-hidden="true">
            <span className="mini-band band-0" style={{ width: `${t1}%` }} />
            <span className="mini-band band-1" style={{ width: `${t2 - t1}%` }} />
            <span className="mini-band band-2" style={{ width: `${100 - t2}%` }} />
            {capped && <span className="mini-raw" style={{ left: `${clampPct(detail.raw_score)}%` }} />}
            <span className={`mini-mark tick-${detail.level}`} style={{ left: `${clampPct(detail.final_score)}%` }} />
          </div>
          <div className="mini-axis num muted small" aria-hidden="true">
            <span>0</span>
            <span style={{ left: `${t1}%` }}>{t1}</span>
            <span style={{ left: `${t2}%` }}>{t2}</span>
            <span style={{ left: "100%" }}>100</span>
          </div>
          <p className="muted small num">{detail.model_version}, {detail.elapsed_ms} ms</p>
        </div>

        <div className="criteria">
          <h2 className="block-title">Năm tiêu chí</h2>
          <table className="criteria-table">
            <tbody>
              {CRITERIA_LABELS.map(([key, label]) => {
                const val = detail.criteria[key];
                const w = detail.weights?.[key] ?? 0;
                const none = val === null || val === undefined;
                return (
                  <tr key={key}>
                    <th scope="row">
                      {label}
                      <span className="weight num">{Math.round(w * 100)}%</span>
                    </th>
                    <td className="bar-cell">
                      <span className="bar">
                        <span className="bar-fill" style={{ width: `${none ? 0 : clampPct(val)}%` }} />
                        <span className="bar-t2" style={{ left: `${t2}%` }} />
                      </span>
                    </td>
                    <td className="num bar-val">{none ? <span className="muted">không xét</span> : fmtScore(val)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="hint">Vạch dọc là ngưỡng Phù hợp ({t2}). Phần trăm là trọng số của tiêu chí trong điểm tổng.</p>
        </div>
      </div>

      <div className="split">
        <div className="block">
          <h2 className="block-title">Có trong CV <span className="num muted">{detail.matched.length}</span></h2>
          {detail.matched.length === 0 ? (
            <p className="muted">Không có kỹ năng nào khớp.</p>
          ) : (
            <ul className="skill-list">
              {detail.matched.map((m, i) => (
                <li key={i}>
                  <span className="skill-name">{m.skill}</span>
                  <span className="skill-kind">{m.type === "must" ? "bắt buộc" : "ưu tiên"}</span>
                  <span className="num skill-score">{fmtScore(m.score * 100)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="block">
          <h2 className="block-title">Còn thiếu <span className="num muted">{detail.missing.length}</span></h2>
          {detail.missing.length === 0 ? (
            <p className="status-ok">Đủ mọi kỹ năng yêu cầu.</p>
          ) : (
            <ul className="skill-list">
              {detail.missing.map((m, i) => (
                <li key={i} className={m.type === "must" ? "is-missing-must" : ""}>
                  <span className="skill-name">{m.skill}</span>
                  <span className="skill-kind">{m.type === "must" ? "bắt buộc" : "ưu tiên"}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <h2 className="block-title section-gap">Đoạn CV khớp nhất với từng đoạn JD</h2>
      <div className="pairs">
        {(detail.best_matches || []).map((bm) => (
          <div className="pair" key={bm.jd_chunk_idx}>
            <blockquote className="pair-jd">{bm.jd_excerpt}</blockquote>
            <blockquote className="pair-cv">{bm.cv_excerpt}</blockquote>
            <span className="pair-sim num" title="Độ tương đồng cosine">{bm.similarity.toFixed(2).replace(".", ",")}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
