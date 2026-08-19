import { useState } from "react";
import { App as AntApp, ConfigProvider, theme } from "antd";
import BodyContent from "./components/BodyContent/BodyContent";
import Workstation from "./features/workspace/Workstation";
import "./styles/tokens.css";
import "./features/workspace/workspace.css";
import "./features/search/search.css";
import "./features/results/results.css";
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

export default function App() {
  const [view, setView] = useState("preview");
  if (view === "legacy") {
    return (
      <div className="legacy-root">
        <BodyContent />
      </div>
    );
  }
  return (
    <ConfigProvider theme={lightTheme}>
      <AntApp>
        <Workstation view={view} onSwitchView={setView} />
      </AntApp>
    </ConfigProvider>
  );
}
