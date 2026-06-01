/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Project theme
        watermelon: {
          DEFAULT: "#f0485f",
          50: "#fff1f3",
          100: "#ffe1e6",
          200: "#ffc9d1",
          400: "#f87489",
          500: "#f0485f",
          600: "#d83452",
          700: "#b32641",
        },
        lemon: {
          DEFAULT: "#fdf7c3",
          50: "#fffef5",
          100: "#fdf7c3",
          200: "#f9ee9a",
          300: "#f3e26b",
        },
        ink: "#2b2b2b",
        muted: "#7a7a7a",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
      },
      boxShadow: {
        soft: "0 6px 24px -8px rgba(43,43,43,0.18)",
        card: "0 2px 12px -4px rgba(43,43,43,0.12)",
      },
      borderRadius: {
        xl2: "1.25rem",
      },
    },
  },
  plugins: [],
};
