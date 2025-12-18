"use client";

import EnrollmentControls from "@/features/control/EnrollmentControls";
import { fetchJSON } from "@/lib/api";
import { Camera } from "@/types";
import React, { useEffect, useState } from "react";

export default function Page() {
  const [cams, setCams] = useState<Camera[]>([]);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  async function load() {
    try {
      setLoading(true);
      setErr("");
      const list = await fetchJSON<Camera[]>("/api/cameras");
      setCams(list);
    } catch (e: any) {
      setErr(e?.message ?? "Failed to load cameras");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="text-lg font-semibold">Employee Enrollment</div>
          <div className="text-xs text-gray-500">
            Select camera + employee, then capture angles and save templates.
          </div>
        </div>
        <button
          className="rounded-md border bg-white px-3 py-1 text-sm"
          onClick={load}
          disabled={loading}
        >
          {loading ? "Refreshing..." : "Refresh Cameras"}
        </button>
      </div>

      {err ? (
        <div className="mb-3 rounded-md border border-red-300 bg-red-50 p-2 text-sm text-red-700">
          {err}
        </div>
      ) : null}

      <EnrollmentControls cameras={cams} />
    </div>
  );
}
