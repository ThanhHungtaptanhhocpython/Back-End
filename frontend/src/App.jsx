import { useEffect, useState } from "react";
import { App as AntApp, ConfigProvider, theme } from "antd";
import Workstation from "./features/workspace/Workstation";
import SettingsPage from "./features/settings/SettingsPage";
import "./styles/tokens.css";
import "./features/workspace/workspace.css";
import "./features/search/search.css";
import "./features/results/results.css";
import "./features/results/temporal.css";
import "./features/selection/selection.css";
import "./features/review/review.css";
import "./features/chat/chat.css";

const lightTheme = {
  algorithm: theme.defaultAlgorithm,
  token: {
    colorPrimary: "#2563eb",
    colorBgBase: "#f8fafc",
    colorBgContainer: "#ffffff",
    colorText: "#0f172a",
    colorBorder: "#e2e8f0",
  },
};

const SETTINGS_HASH = "#/settings";

function useSettingsRoute() {
  const [open, setOpen] = useState(
    () => typeof window !== "undefined" && window.location.hash.startsWith(SETTINGS_HASH),
  );
  useEffect(() => {
    const sync = () => setOpen(window.location.hash.startsWith(SETTINGS_HASH));
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);
  return [
    open,
    () => {
      if (window.location.hash.startsWith(SETTINGS_HASH)) window.location.hash = "";
    },
  ];
}

export default function App() {
  const [settingsOpen, closeSettings] = useSettingsRoute();

  return (
    <ConfigProvider theme={lightTheme}>
      <AntApp>
        {settingsOpen ? <SettingsPage onClose={closeSettings} /> : <Workstation />}
      </AntApp>
    </ConfigProvider>
  );
}
