# Next dev static asset 404 원인 정리

## 증상

- `http://127.0.0.1:3000` root HTML은 200
- 화면은 무스타일 HTML처럼 보임
- `_next/static/...css` 또는 JS asset이 404

## 원인

Next dev server가 떠 있는 상태에서 `npm run build`를 실행하면 `.next` 산출물과 build id가 바뀔 수 있다. 이때 기존 dev server가 계속 살아 있으면 브라우저가 받은 HTML의 static 경로와 실제 dev server가 제공하는 `_next/static` 경로가 어긋난다.

즉 CSS 자체가 깨진 것이 아니라, dev server lifecycle과 `.next` 산출물의 기준이 어긋난 상태다.

## 현재 확인

- frontend root: 200
- HTML 내 stylesheet asset: 200
- 브라우저 화면 스타일 적용 확인

## 복구 원칙

1. `npm run build` 후에는 dev server를 재시작한다.
2. readiness는 root 200만 보지 말고 CSS asset 200까지 확인한다.
3. stale dev server가 의심되면 3000 포트 node 프로세스를 정리한 뒤 다시 띄운다.
4. `.next` 삭제는 기본 조치가 아니라, 재시작으로 복구되지 않을 때의 수동 옵션으로 둔다.
