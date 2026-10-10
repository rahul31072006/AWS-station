import type { Metadata } from 'next';
import './globals.css';
import { StationProvider } from './station-context';

export const metadata: Metadata = {
  title: 'SkyGuard AI | Weather Intelligence Command Center',
  description: 'AI-powered AWS anomaly detection and sensor intelligence.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><StationProvider>{children}</StationProvider></body></html>;
}
