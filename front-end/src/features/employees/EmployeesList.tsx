import { Card } from "@/components/ui/Card";

import { Employee } from "@/types";

export default function EmployeesList({
  employees,
}: {
  employees: Employee[];
}) {
  return (
    <Card title="Employees">
      <p className="text-sm text-gray-500">Total: {employees.length}</p>
      <ul className="mt-3 space-y-1">
        {employees.map((e) => (
          <li key={e.id}>
            <b>{e.name}</b>{" "}
            <span className="text-sm text-gray-500">({e.id})</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}
