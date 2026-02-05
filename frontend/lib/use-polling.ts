"use client";

import { useEffect, useState, useCallback, useRef } from "react";

interface UsePollingResult<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number = 60_000,
): UsePollingResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const doFetch = useCallback(async (isInitial: boolean) => {
    if (isInitial) setLoading(true);
    try {
      const result = await fetcherRef.current();
      setData(result);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    } finally {
      if (isInitial) setLoading(false);
    }
  }, []);

  useEffect(() => {
    doFetch(true);
    const id = setInterval(() => doFetch(false), intervalMs);
    return () => clearInterval(id);
  }, [doFetch, intervalMs]);

  return { data, loading, error };
}
