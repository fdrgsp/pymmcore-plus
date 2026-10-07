# Autofocus

Autofocus happens inside an acquisition when an event carries a
[`useq.HardwareAutofocus`](https://pymmcore-plus.github.io/useq-schema/schema/event/)
or `useq.SoftwareAutofocus` action, which
[`useq.AxesBasedAF`](https://pymmcore-plus.github.io/useq-schema/schema/hardware_autofocus/)
and `useq.SoftwareAxesBasedAF` insert on the axes you choose.

Every attempt reports an `AutofocusResult`, both as the return value of a software
routine and on the
[`autofocusFinished`][pymmcore_plus.mda.events.PMDASignaler.autofocusFinished] signal.

::: pymmcore_plus.autofocus.AutofocusResult
