"""initial placeholder

Revision ID: 0001_initial_placeholder
Revises:
Create Date: 2026-06-04 00:00:00.000000

CP236d — Alembic 골격이 실제로 revision을 만들 수 있다는 증명용 placeholder.
upgrade()/downgrade() 본문은 pass. 결제 후 실 스키마 마이그레이션은 별도 CP.
"""

from typing import Sequence, Union

# from alembic import op
# import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_initial_placeholder"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
