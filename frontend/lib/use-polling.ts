"use client";

import { useEffect, useRef, useState } from "react";

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
  const mountedRef = useRef(false);
  const requestIdRef = useRef(0);

  useEffect(() => {
    fetcherRef.current = fetcher;
  }, [fetcher]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function doFetch(isInitial: boolean): Promise<void> {
      if (isInitial) setLoading(true);
      const requestId = ++requestIdRef.current;
      try {
        const result = await fetcherRef.current();
        if (cancelled || !mountedRef.current || requestId !== requestIdRef.current) {
          return;
        }
        setData(result);
        setError(null);
      } catch (e) {
        if (cancelled || !mountedRef.current || requestId !== requestIdRef.current) {
          return;
        }
        setError(e instanceof Error ? e : new Error(String(e)));
      } finally {
        if (isInitial && !cancelled && mountedRef.current && requestId === requestIdRef.current) {
          setLoading(false);
        }
      }
    }

    void doFetch(true);
    const id = window.setInterval(() => {
      void doFetch(false);
    }, intervalMs);

    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [intervalMs]);

  return { data, loading, error };
}
