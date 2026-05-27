// Next.js 14 의 PostCSS 처리 명시 설정.
// devDependency 에 tailwindcss 가 남아있지만 globals.css 에서 사용하지 않는다.
// Tailwind plugin 자동 감지를 끄고 autoprefixer 만 활성화한다.
// dev 모드에서 _next/static/css/app/layout.css 가 404 나는 문제 해결.
module.exports = {
  plugins: {
    autoprefixer: {},
  },
};
