"""add connector deliveries, external events, event rules, and turn source

Revision ID: e5a7b3c91d40
Revises: c8f4a912d3e7
Create Date: 2026-10-25 14:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e5a7b3c91d40'
down_revision: Union[str, None] = 'c8f4a912d3e7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('turns', schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                'source',
                sa.String(length=16),
                nullable=False,
                server_default='user',
            )
        )

    op.create_table(
        'connector_deliveries',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('principal_id', sa.String(length=32), nullable=False),
        sa.Column('source', sa.String(length=64), nullable=False),
        sa.Column('event_type', sa.String(length=128), nullable=False),
        sa.Column('dedupe_key', sa.String(length=255), nullable=False),
        sa.Column('payload', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(length=24), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ['principal_id'], ['principals.id'], ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('dedupe_key'),
    )
    with op.batch_alter_table('connector_deliveries', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_connector_deliveries_principal_id'),
            ['principal_id'],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f('ix_connector_deliveries_source'), ['source'], unique=False
        )
        batch_op.create_index(
            batch_op.f('ix_connector_deliveries_status'), ['status'], unique=False
        )

    op.create_table(
        'external_events',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('principal_id', sa.String(length=32), nullable=False),
        sa.Column('delivery_id', sa.String(length=32), nullable=True),
        sa.Column('source', sa.String(length=64), nullable=False),
        sa.Column('event_type', sa.String(length=128), nullable=False),
        sa.Column('dedupe_key', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=24), nullable=False),
        sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=False),
        sa.Column('raw', sa.JSON(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('decision', sa.String(length=16), nullable=True),
        sa.Column('decided_by', sa.String(length=128), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ['principal_id'], ['principals.id'], ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['delivery_id'], ['connector_deliveries.id'], ondelete='SET NULL'
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('dedupe_key'),
    )
    with op.batch_alter_table('external_events', schema=None) as batch_op:
        for column in (
            'principal_id', 'source', 'event_type', 'status',
            'occurred_at', 'processed_at',
        ):
            batch_op.create_index(
                batch_op.f(f'ix_external_events_{column}'), [column], unique=False
            )

    op.create_table(
        'event_rules',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('principal_id', sa.String(length=32), nullable=False),
        sa.Column('rule_id', sa.String(length=64), nullable=False),
        sa.Column('match', sa.JSON(), nullable=False),
        sa.Column('when', sa.Text(), nullable=True),
        sa.Column('action', sa.String(length=16), nullable=False),
        sa.Column('urgent', sa.Boolean(), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('note', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ['principal_id'], ['principals.id'], ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'principal_id', 'rule_id', name='uq_event_rule_identity'
        ),
    )
    with op.batch_alter_table('event_rules', schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f('ix_event_rules_principal_id'),
            ['principal_id'],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table('event_rules', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_event_rules_principal_id'))
    op.drop_table('event_rules')

    with op.batch_alter_table('external_events', schema=None) as batch_op:
        for column in (
            'principal_id', 'source', 'event_type', 'status',
            'occurred_at', 'processed_at',
        ):
            batch_op.drop_index(batch_op.f(f'ix_external_events_{column}'))
    op.drop_table('external_events')

    with op.batch_alter_table('connector_deliveries', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_connector_deliveries_status'))
        batch_op.drop_index(batch_op.f('ix_connector_deliveries_source'))
        batch_op.drop_index(batch_op.f('ix_connector_deliveries_principal_id'))
    op.drop_table('connector_deliveries')

    with op.batch_alter_table('turns', schema=None) as batch_op:
        batch_op.drop_column('source')
