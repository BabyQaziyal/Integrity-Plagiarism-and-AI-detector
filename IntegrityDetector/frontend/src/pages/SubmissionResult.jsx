import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import ResultView from "../components/ResultView";
import { ErrorState, Skeleton } from "../components/ui";

export default function SubmissionResult() {
  const { id } = useParams();
  const [submission, setSubmission] = useState(null);
  const [error, setError] = useState(null);

  const load = () => {
    setSubmission(null);
    setError(null);
    api.getSubmission(id).then(setSubmission).catch(setError);
  };
  useEffect(load, [id]);

  return (
    <div className="space-y-4">
      <Link to="/" className="inline-flex items-center gap-1 text-sm font-semibold text-watermelon hover:underline">
        ← New analysis
      </Link>

      {error ? (
        <ErrorState
          title="Couldn’t load this analysis"
          message={error.offline ? "The backend isn’t running." : error.message}
          onRetry={load}
        />
      ) : !submission ? (
        <div className="space-y-4">
          <Skeleton className="h-40 w-full" />
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Skeleton className="h-28 w-full" />
            <Skeleton className="h-28 w-full" />
            <Skeleton className="h-28 w-full" />
          </div>
          <Skeleton className="h-64 w-full" />
        </div>
      ) : (
        <ResultView submission={submission} />
      )}
    </div>
  );
}
