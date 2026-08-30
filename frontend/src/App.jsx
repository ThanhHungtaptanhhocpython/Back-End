import { App as AntApp, ConfigProvider, theme } from "antd";
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
  return (
    <ConfigProvider theme={lightTheme}>
      <AntApp>
        <Workstation />
      </AntApp>
    </ConfigProvider>
  );
}
