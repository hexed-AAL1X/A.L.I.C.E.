/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'alice-ink': '#121214',
        'alice-ink-soft': '#1a1a1d',
        'alice-muted': '#8a8a90',
        'alice-line': '#2c2c30',
        'alice-soft': '#ececee',
        'alice-paper': '#f4f5f8',
        'alice-accent': '#3a3a40',
        'alice-dark': '#0a0a0b',
        'primary-blue': '#3a3a40',
        'light-blue': '#9a9aa0',
        'deep-blue': '#0a0a0b',
        'blue-gray': '#ececee',
        'text-dark': '#121214',
        'text-gray': '#5c5c64',
      },
      fontFamily: {
        sans: ['Outfit', 'system-ui', 'sans-serif'],
        display: ['Source Serif 4', 'Georgia', 'serif'],
      },
      letterSpacing: {
        brand: '0.08em',
      },
    },
  },
  plugins: [],
}
