/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      boxShadow: {
        "custom": '0 0px 13px -8px '
      },
      screens: {
        'below-1488': {'max': '1488px'}, // Custom media query for width < 1488px
      },
    },
  },
  plugins: [],
}

