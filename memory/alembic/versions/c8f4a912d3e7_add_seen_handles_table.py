"""add seen_handles table

Revision ID: c8f4a912d3e7
Revises: 55d0f1042d65
Create Date: 2026-10-25 00:25:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c8f4a912d3e7'
down_revision: Union[str, None] = '55d0f1042d65'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'seen_handles',
        sa.Column('message_handle', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('message_handle'),
    )
    with op.batch_alter_table('seen_handles', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_seen_handles_created_at'),
            ['created_at'],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table('seen_handles', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_seen_handles_created_at'))

    op.drop_table('seen_handles')
