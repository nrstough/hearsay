import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        canvas: "#faf6f0",
        nsa: {
          blue: "#005493",
          navy: "#00254b",
          dark: "#001730",
          light: "#0b66b2",
          gold: "#c37530",
          bronze: "#9e591e",
          amber: "#ffb800",
        },
      },
      borderRadius: {
        "2xl": "18px",
        "3xl": "24px",
        "4xl": "32px",
      },
      fontFamily: {
        sans: ["var(--font-satoshi)", "-apple-system", "BlinkMacSystemFont", "'Segoe UI'", "Roboto", "sans-serif"],
        display: ["var(--font-satoshi)", "-apple-system", "BlinkMacSystemFont", "'Segoe UI'", "Roboto", "sans-serif"],
        satoshi: ["var(--font-satoshi)", "sans-serif"],
        inter: ["var(--font-satoshi)", "sans-serif"],
        mono: ["var(--font-satoshi)", "sans-serif"],
        dotted: ["var(--font-doto)", "'Doto'", "monospace"],
        numbers: ["var(--font-doto)", "'Doto'", "monospace"],
      },
      boxShadow: {
        card: "0 2px 12px -2px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02)",
        "card-hover": "0 8px 24px -4px rgba(0, 0, 0, 0.07), 0 2px 6px rgba(0, 0, 0, 0.03)",
        glass: "0 20px 40px -15px rgba(0, 0, 0, 0.2)",
        popover: "0 12px 32px -8px rgba(0, 0, 0, 0.24)",
      },
    },
  },
  plugins: [],
};
export default config;
