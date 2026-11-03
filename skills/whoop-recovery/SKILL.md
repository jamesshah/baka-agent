---
name: whoop-recovery
description: How to talk about Whoop recovery, sleep, and strain events and questions.
tools: [query_events]
triggers: [whoop, recovery, sleep, hrv, strain, workout, resting heart rate, sleep debt]
---
Use when a Whoop event arrives or the user asks about their recovery, sleep, or training.

Reading the numbers:
- Recovery: green is 67-100%, yellow 34-66%, red 0-33%. HRV and resting heart rate
  matter more as trends than as single values; compare against the past week with
  query_events before calling something unusual.
- Sleep performance is hours slept versus hours needed. Sleep debt over an hour is
  worth mentioning; under that, skip it.
- "Still calibrating" means the first few weeks of data; soften any conclusions.

Writing the message:
- Red recovery: say it plainly, suggest an easier day (zone 2, mobility, or rest),
  and an earlier bedtime tonight. If you can see a hard workout planned today,
  suggest moving or easing it.
- Short sleep: lead with the hours, mention debt if over an hour, suggest a nap
  window or an earlier night. Don't lecture.
- Green after a rough stretch: a quick positive note is enough.
- One or two sentences. Use the numbers, round them, and never paste raw JSON.
- Not medical advice: if HRV or resting heart rate is far off the user's usual
  range for several days, suggest checking in on how they feel, nothing more.
