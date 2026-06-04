"""Lens repositories.

현재: supabase-py REST 전용 (ORM 없음). 장차 ORM repository를 추가할 때는
응답에 필요한 관계를 반드시 eager load (selectinload/joinedload)하고,
루프 내 행별 쿼리 (N+1)를 금지한다. 상세 규약: docs/db_repository_guide.md (CP236c).
"""
