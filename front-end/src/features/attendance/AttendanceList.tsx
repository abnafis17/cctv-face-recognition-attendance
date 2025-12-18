import { Card } from "@/components/ui/Card";

import { AttendanceRow } from "@/types";

export default function AttendanceList({
  attendance,
}: {
  attendance: AttendanceRow[];
}) {
  return (
    <Card title="Recent Attendance">
      <p className="text-sm text-gray-500">
        Showing latest {attendance.length}
      </p>

      <div className="mt-3 space-y-2">
        {attendance.map((a) => (
          <div key={a.id} className="rounded-lg border border-gray-200 p-3">
            <div className="flex items-start justify-between gap-3">
              <div>
                <b>{a.name}</b>{" "}
                <span className="text-sm text-gray-500">({a.employeeId})</span>
              </div>
              <div className="text-sm text-gray-600">
                {new Date(a.timestamp).toLocaleString()}
              </div>
            </div>

            <div className="mt-1 text-xs text-gray-500">
              Camera: {a.cameraId ?? "N/A"} | Confidence:{" "}
              {typeof a.confidence === "number"
                ? a.confidence.toFixed(3)
                : "N/A"}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
