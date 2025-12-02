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
        // Paleta ElevenLabs
        'primary-blue': '#1A4FBF',
        'light-blue': '#68A8FF',
        'deep-blue': '#0B2A66',
        'blue-gray': '#E8F0FF',
        'text-dark': '#0D0D0D',
        'text-gray': '#555555',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Sora', 'system-ui', 'sans-serif'],
      },
      letterSpacing: {
        'brand': '0.04em',
      },
    },
  },
  plugins: [],
}
