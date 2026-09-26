# HEARSAY: Master Doc (HackGT 13)

> Exported Sep 25, 2026 from `HEARSAY Master Doc (HackGT 13).pdf` (Claude Docs export, dated Sep 24)
> with `pdftotext -layout`; page headers and footers removed, otherwise verbatim. Layout text is
> kept in a fenced block so the tables survive. `docs/plan.md` supersedes this doc where they differ.

```text
    Sep 24, 2026        · @Nathan Stough

  Status and decisions
  HEARSAY is the team's HackGT 13 project as of Thu Sep 24; Reach is shelved. The build is a
  listening head that turns toward whoever is talking and calls out AI-generated voices, plus
  the NSA CSV submission from the same model.

  Decided

       Project: HEARSAY with hardware (the listening head), replacing Reach.
       One model powers both deliverables: the NSA test-set CSV and the live head.
       Hardware reuses the Reach pointer idea: pan-tilt servos, a camera and a laser, plus a
       mic array.
       No hosted API on the live demo path; everything runs locally on a laptop.
       The rig for mass replay recording is cut as busy work; replay training is a short Friday-
       night session instead.

  Open tonight

       Track: The Shipyard (hardware) is the plan, if a hardware-track entry can also submit to
       NSA HEARSAY
       Mic array: buy a ReSpeaker-style USB array, or build from I2S MEMS mics
       GPU access for the weekend (needed only for fine-tuning, not the baseline)
       Post the Q&A questions (see Rules and open questions)
       Tell the team; each owner reads their section before Friday

  Overview
  HEARSAY is a tabletop listening head that turns toward whoever is speaking, decides in
  real time whether the voice is real or AI-generated, and locks onto a fake with a red laser
  dot.

       Event: HackGT 13, Georgia Tech (Klaus), Sep 25 to 27, 2026. Four-person team.
       User story: voice-clone scam calls. A cloned voice of a grandchild or boss asks for
       money, and the listener can't tell. HEARSAY is a second set of ears in the room.

       Pitch in one line: two phones play the same voice; one is real and one is cloned. The
       head swings between them, stays green on the real one, and pins the fake in red.
       Pitch statistic: voice-clone scam losses or incident counts from an official source (FTC
       or FBI IC3). Confirm the figure before it goes on a slide.
       What is actually the product: the detection model and the live scoring pipeline. The
       head makes it visible; the NSA test-set score proves it works.
       Track: The Shipyard (hardware), pending the Q&A answer on stacking with NSA.
       Prize stack: Shipyard track, Best Overall, NSA HEARSAY, Notability, Create-X, plus
       Meta as a maybe (see Sponsor challenges).
       Why this can win: a measurable result (the NSA score), a demo judges understand
       from across the room, and a timely problem.

  The NSA challenge
  NSA wants a system that takes any audio clip and outputs a 0 to 100% likelihood that the
  voice is AI-generated, scored against a hidden test set.

       Sponsor: NSA, framed as their Multimedia Forensics and Authentication Team.
       Input: any audio file (WAV, MP3, M4A and others); the system handles conversion
       itself.
       Output: a synthetic likelihood score per clip; optionally the manipulation type (text-to-
       speech, voice cloning, splicing).
       Data: a labeled training set and a held-out test set, provided at the event.
       Submission: a CSV of predictions on the test set, scored automatically.
       Allowed tools: cloud services, open-source LLMs, speech models and signal-
       processing toolkits. They care about how techniques are combined.
       Prizes (per team member): Sonos Era 100 (1st), Audio-Technica AT2040 USB mic (2nd),
       JBL Flip 5 (3rd).

  Unknown until the event: dataset size, which generators made the fakes, the scoring
  metric, whether manipulation types are labeled, and whether outside data is allowed.

  System architecture
  Two pipelines share one model: an offline batch path that writes the NSA CSV, and a live
  path that listens, aims and alerts.

     flowchart LR
       subgraph Offline
         T[NSA test clips] --> L[Loader<br/>16 kHz mono]
         L --> M[Detector ensemble]
         M --> C[CSV]
       end
       subgraph Live
         A[Mic array] --> V[Voice activity<br/>+ direction]
         V --> H[Head aims<br/>at source]
         A --> W[2 to 4 s windows]
         W --> M2[Same detector]
         M2 --> S[Smoothed score<br/>per source]
         S --> H
         S --> D[Dashboard]
       end

  Data contract (agree on this Friday night before coding)

       Audio front end outputs {t_start, t_end, angle_deg, audio_window} whenever
       someone is speaking.
       The detector takes a 16 kHz mono window and returns {p_synthetic,
       per_detector_scores} .

       The tracker outputs {source_id, angle_deg, score_smoothed, verdict} and sends
       it to the head and the dashboard.
       The head driver takes {pan_deg, tilt_deg, laser, color} and returns {ok,
       error} .

  Each owner can build against fake data on day one: the model against WAV files, the head
  against a scripted angle, the dashboard against a recorded JSON log.

  Team roles
  Each person owns one layer and builds against the data contract, so nobody waits on
  anyone else.

     Person                       Owns                               First deliverable (Fri night)

     Nathan                       Detector ensemble, validation      Baseline CSV scored on a held-
                                  split, NSA CSV, streaming scorer   out split

     Person                          Owns                                    First deliverable (Fri night)

     EE teammate                     Mic array, direction finding            Angle estimate that tracks a clap
                                     (TDOA), voice activity detection,       across the table
                                     servo and laser power

     CS-hardware                     Pan-tilt head, servo driver, aiming,    Head turns to a commanded
     teammate                        camera snap-to-source, enclosure        angle and back

     Frontend                        Dashboard, per-second timeline,         Dashboard replaying a recorded
     teammate                        demo video from hour one,               JSON log
                                     Devpost

  Handoffs that matter: EE to CS-hardware is the angle feed; Nathan to everyone is
  p_synthetic per window; everyone to frontend is the JSON log.

  Hardware: the listening head
  The head is a pan-tilt mount carrying a camera and a laser, fed by a mic array that
  estimates where the voice is coming from. Pan does the tracking; tilt only has to aim down
  at the table.

  Parts

     Part                         Choice                                    Notes

     Mic array,                   ReSpeaker-style USB 4-mic array           Plug and play, has onboard
     option A                                                               direction estimate; check it can
                                                                            arrive by Friday

     Mic array,                   2 to 4 INMP441 I2S MEMS mics on           Cheap and real EE work; 2 mics are
     option B                     an ESP32                                  enough for pan-only

     Servos                       2x hobby servos (SG90 enough;             Pan and tilt; the load is light
                                  MG90S or MG996R if the desk has
                                  them)

     Mount                        Pan-tilt bracket kit, a quick print, or   CAD owner's call
                                  cut acrylic

     Camera                       USB webcam on the tilt bracket,           Snaps the aim onto the phone or
                                  bore-sighted with the laser               person

     Part                         Choice                              Notes

     Pointer                      Class 2 laser module (under 1 mW)   Red lock on fakes, green glow on
                                  plus an RGB LED                     real voices

     Driver                       Arduino Uno or Nano (or PCA9685     ESP32 can do both mics and servos
                                  board) over USB serial              in option B

     Power                        Separate 5V 2A supply for servos,   Keeps servo spikes out of the
                                  shared ground                       audio

     Optional                     Servo-driven analog gauge (needle   Cheap, readable from across the
                                  from REAL to SYNTHETIC)             room

  Direction finding
  Two mics hear the same voice a tiny bit apart in time. The delay gives the angle:

  θ = arcsin(c · τ / d)

       θ (degrees): angle of the source from straight ahead, measured from the line
       perpendicular to the mic pair
       c (m/s): speed of sound, a constant, about 343 m/s at room temperature
       τ (s): time difference of arrival between the two mics, measured with GCC-PHAT
       cross-correlation
       d (m): spacing between the two mics, set by the build

  With d = 0.15 m, the largest possible τ is 0.15 / 343 ≈ 0.44 ms, about 21 samples at a 48 kHz
  sample rate. That gives roughly 3° steps straight ahead on paper; expect 10 to 20° in a
  noisy, echoing hall. That is why the camera snaps the final aim onto the nearest phone.

  Noise handling: only estimate the angle while voice activity is detected, take the median
  over each window, and ignore angles that jump more than about 30° in one step.

  Aiming
   1. The angle estimate swings the pan servo toward the source.
   2. The camera finds the nearest phone or face near frame center (a standard detector is
      enough).
   3. The head nudges until the target is centered, then holds.
   4. If the smoothed score crosses the threshold, the laser and red LED turn on; below it,
      green.

  Safety: class 2 laser only, a software tilt limit so it never points above the table, and the
  dot aimed at the phone, not at people's faces.

  Hardware tiers

     Tier      What it adds                                 Done when

     H0        Head turns to a commanded angle; mic         Head points within 20° of a clap, 8 of 10
               array reports an angle                       times

     H1        Live tracking of voices                      Head follows two phones taking turns

     H2        Score on each source; red and green          Head stays green on the real voice and
               verdicts                                     turns red on the clone

     H3        Camera snap-to-source and laser lock         Dot lands on the fake phone, not the
                                                            space beside it

     H4        Enclosure, gauge, cable management           Looks like a product on the demo table

  Model ladder
  Submit a valid CSV within the first few hours and never be without one; every rung after
  that is an improvement measured on a held-out split.

     Tier      What it adds                        Done when                            Effort

     M0        Loader (any format to 16 kHz        Split exists; loader handles         Low
               mono, chunk long clips) and a       every file in the training set
               validation split held out by
               generator or speaker

     M1        Frozen WavLM or XLS-R               First CSV submitted; validation      Low
               embeddings, pooled, into logistic   score recorded
               regression

     M2        Hand-crafted features: LFCC or      Validation score beats M1 alone      Low
               MFCC stats, spectral flatness,      when combined
               high-frequency energy cutoff,
               pause patterns

     Tier      What it adds                       Done when                           Effort

     M3        Off-the-shelf detector scores      Adds signal on validation, or       Very low
               (for example an AASIST3            gets dropped
               checkpoint) as extra features

     M4        Stacked gradient-boosted           Best validation score so far; CSV   Medium
               ensemble with calibrated 0 to      resubmitted
               100% output (Platt or isotonic)

     M5        Fine-tuned SSL model with an       Beats M4 on validation              High
               AASIST head (needs GPU)

     M6        Manipulation-type head, if types   Per-type accuracy reported          Low once M4
               are labeled                                                            exists

  Live scoring
       Score 2 to 4 second windows while voice activity is detected; short windows are less
       accurate, so never alert on one window.
       Smooth per source with an exponential moving average; alert only after the smoothed
       score stays above threshold for about 3 windows.
       The live path uses the fastest model that's good enough (likely M1 or M4 without M5),
       since it has to keep up in real time.

  Replay training (Friday night, about an hour)
       Everything the head hears comes through the air, which is known to hurt detectors
       badly; in one study replay raised a top model's error rate from 4.7% to 18.2%.
       Fix: play a few hundred training clips (real and fake) through two or three phones at
       the actual mic array, record them, and add them to training with their original labels.
       Also augment with phone codecs (AMR, Opus) in software.
       Keep a clean-only validation score too; the NSA test set is probably clean audio, so the
       CSV model must not get worse on it.

  Model rules: validate by generator, not by clip; log every CSV with its validation score; cap
  any rung at a few hours.

  Dashboard and UI
  The dashboard explains what the head is doing; judges watch the head, then look at the
  screen to see why.

       Live view: a top-down table diagram showing each detected source at its angle,
       colored green or red, with the head's current aim.
       Timeline: the waveform of the last 30 seconds, each second shaded by its synthetic
       score, so a splice lights up in the middle of real speech.
       Why panel: per-detector scores for the current source (SSL model, spectral artifacts,
       pause patterns, off-the-shelf detector) and the manipulation type if M6 exists.
       Upload mode: drop in any audio file and get the same timeline and score. This is the
       NSA-style use and a fallback if the head fails.
       Replay mode: play back a recorded JSON log, for the demo video and for building the
       UI before the hardware works.

  Rule: the dashboard never blocks the head. If the UI crashes, the head keeps working.

  Sponsor challenges
  The switch from Reach changes which sponsor prompts fit: NSA becomes the core,
  SpaceXAI drops, and Meta becomes worth a look.

     Challenge                    Verdict          How it attaches               Cost

     NSA HEARSAY                  Core             The CSV from the model        This is the
                                                   ladder; the head is the       project
                                                   product around it

     Notability                   Do               Use Notability Pro for        About 15
                                                   planning or to transcribe     minutes
                                                   practice runs; Devpost tag,
                                                   short note, 2 screenshots

     Create-X                     Do               Interest checkbox; scam-      One click
                                                   call protection is a
                                                   plausible startup pitch

     Challenge                    Verdict              How it attaches                   Cost

     Meta                         Maybe, reread the    It is judged on                   Only framing, if
                                  prompt               strengthening human               it fits
                                                       connection. Protecting
                                                       trust in calls with family is a
                                                       better fit than Reach was

     SpaceXAI                     Probably skip        Needs a Cursor and Grok           A few hours for
                                                       build; no natural Grok role       a small prize
                                                       on the live path, and the
                                                       public-health framing is
                                                       weaker

     NSA Packet Pursuit,          Skip                 CTF and reverse                   n/a
     Codebreaker                                       engineering; Codebreaker
                                                       is for later

     Impiricus, Visa              Skip unless Visa's   Reread Visa's wording for a       n/a
                                  prompt covers        fraud angle before ruling it
                                  fraud                out

  Rules and open questions
  The one question that shapes the whole plan is whether a hardware-track project can also
  submit to the NSA challenge.

  Confirmed (pre-event packet)

       One track per team.
       The hardware desk lends servos; not returning borrowed hardware means a
       permanent HexLabs ban.
       Sponsor challenges stack on top of the one project.

  Assumed from HackGT 12 rules (check the packet)

       No coding before 8 pm Friday.
       Credit every public model and framework used.
       One project per team.

  Open questions (post in the HackGT 13 Q&A thread on Discord)

       Can a Shipyard (hardware) project also submit to NSA HEARSAY, or does HEARSAY
       need its own track?
       Is outside training data (MLAAD, ASVspoof, In-the-Wild) allowed for HEARSAY?
       What metric scores the CSV, and is there a submission limit?
       Are pretrained model downloads allowed before 8 pm Friday (weights only, no code)?
       What servos, mics or microcontrollers does the hardware desk have?
       Are laser pointers allowed at the demo table (class 2, aimed at the table)?
       Do AI coding tools need to be disclosed, and how?

  Risks and mitigations
  The model side is low risk because a baseline scores in hours; the risks live in the live
  demo, where audio passes through the air in a loud room.

     Risk                         Likelihood   What it costs          Mitigation

     Replay through phone         High         Head calls the real    Friday-night replay
     speakers wrecks live                      voice fake on stage    training; pick demo clips
     accuracy                                                         the live model gets right
                                                                      every time in rehearsal

     Hall noise and echo          High         Head points at the     Voice activity gating,
     break direction finding                   wall                   median filtering, camera
                                                                      snap-to-source, sources
                                                                      placed well apart

     Can't submit to both         Medium       One prize pool lost    Ask tonight; if forced to
     Shipyard and NSA                                                 choose, pick the track
                                                                      and still submit the CSV
                                                                      if allowed

     Model overfits to known      Medium       Good validation,       Hold out by generator;
     generators                                bad NSA score          stack diverse detectors

     Mic array doesn't arrive     Medium       Option A gone          Option B with I2S mics,
     by Friday                                                        or two USB mics for pan-
                                                                      only

     Risk                         Likelihood      What it costs        Mitigation

     Servo noise gets into the    Medium          Worse scores while   Separate servo supply;
     audio                                        the head moves       score only when the
                                                                       head is still

     No GPU                       Medium          No M5 fine-tuning    M1 to M4 run on a laptop
                                                                       CPU

     Wi-Fi drops during           High            Anything hosted      Everything local on the
     judging                                      stalls               live path

     Scope creep                  High for this   Nothing solid by     Tiers behind flags, time
                                  team            Sunday               boxes, freeze Sunday
                                                                       morning

  Fallback demo: if the head fails, run upload mode on the dashboard with the same two
  clips; the model and the NSA score still stand.

  Weekend plan, demo and submission
  The demo is two phones, one real voice and one clone, and a head that tells them apart
  live; everything else supports that moment.

  Tonight (Thu)

       Tell the team; share this doc
       Post the Q&A questions
       Order or confirm: mic array (A or B), 2 servos, pan-tilt bracket, class 2 laser, RGB LED,
       Arduino or ESP32, 5V supply, USB webcam
       Find GPU access if any (a teammate's machine or a cloud credit)
       Pick two or three phones for replay training and the demo
       Make or find a cloned-voice clip and its real source for the demo (a teammate's own
       voice, with their consent)

  Weekend timeline

     When                         Model (Nathan)                Hardware (EE + CS-         Frontend
                                                                hardware)

     Fri evening                  Data contract agreed; M0      Contract agreed; head      Contract agreed;
                                  loader and split; coding at   wired and moving           dashboard shell
                                  8 pm

     Fri night                    M1 CSV submitted; replay      H0: angle estimate plus    Replay mode on a
                                  training session              commanded aim              fake log

     Sat morning                  M2 and M3 features            H1: live tracking of two   Live view hooked
                                                                phones                     to the tracker

     Sat                          M4 ensemble, calibrated;      H2: red and green          Timeline and why
     afternoon                    CSV resubmitted; live         verdicts                   panel
                                  scorer

     Sat night                    M5 if GPU; M6 if labels       H3: camera snap and        Upload mode;
                                  exist                         laser lock; H4 enclosure   start demo video

     Sun morning                  Freeze; final CSV             Freeze; rehearse 10+       Freeze; finish
                                                                times                      video

     Sun before                   Devpost technical section     Return borrowed            Devpost, sponsor
     deadline                                                   hardware                   items

  Demo script (about 2 minutes)

   1. The problem: a cloned voice calls asking for money, and people can't tell.
   2. Two phones play the same voice in turns. The head swings to each; green on the real
      one, red and a laser lock on the clone.
   3. The dashboard shows why: the timeline and per-detector scores.
   4. Upload mode on a judge's choice of clip, if they want to test it.
   5. The NSA test-set score, and one line on where this goes next (a call-screening device
       or app).

  Demo table rules: same phones, same positions and same clips as rehearsal; the room as
  quiet as it gets; the recorded video on standby.

  Submission checklist

       Final NSA CSV submitted
       Devpost in the chosen track, crediting every model and dataset

       Demo video
       Notability tag, note and 2 screenshots
       Create-X interest checkbox
       Meta framing, if the prompt fits
       All borrowed hardware returned

  Related docs and sources
  This doc supersedes the earlier HEARSAY research doc; the Reach docs are kept for
  reference.

       HEARSAY: NSA Audio Deepfake Challenge (earlier research, approaches table)
       Reach: Master Doc (shelved; its pan-tilt aiming work carries over)

  Sources

       Audio Deepfake Detection: A Survey
       Spoofing and deepfake detection using wav2vec 2.0 and data augmentation (Tak et al.)
       Codecfake dataset and baselines
       iWAX: Wav2vec-AASIST-XGBoost
       MLAAD dataset
       In-the-Wild dataset
       AASIST3 on Hugging Face
       AASIST official code
```
