'use client';

import { createContext, useContext, useState } from 'react';

export type StationSelection = {
  selectedStationId: string | null;
  selectionMode: 'overall' | 'station';
  setSelectedStationId: (stationId: string | null) => void;
};

const StationContext = createContext<StationSelection | null>(null);

export function StationProvider({ children }: { children: React.ReactNode }) {
  const [selectedStationId, setSelectedStationId] = useState<string | null>(null);
  return <StationContext.Provider value={{
    selectedStationId,
    selectionMode: selectedStationId ? 'station' : 'overall',
    setSelectedStationId,
  }}>{children}</StationContext.Provider>;
}

export function useStationSelection() {
  const context = useContext(StationContext);
  if (!context) throw new Error('useStationSelection must be used inside StationProvider');
  return context;
}
