import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./styles/tokens.css";
import "./styles/theme.css";
import "./styles/global.css";
import "./styles/components.css";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("#root element not found");
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
