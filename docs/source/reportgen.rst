Report Generation
=================================

After running SETs, a final report is generated of the instance. Reporters generate a JSON file that contains
logs of the executed SET(s). From the JSON file, a human-readable HTML file is further generated, that includes
a summary of the executed SET(s), as well as suggestions for possible actions to take if vulnerabilities were found
in the evaluated target.

.. toctree::
   :maxdepth: 3

   avise.reportgen.reporters

Regenerating HTML from JSON
---------------------------

Any JSON report can be rendered again as HTML, for example after the HTML
renderers have been updated:

.. code-block:: bash

   python -m avise.reportgen.render path/to/report.json [-o report.html]

Several JSON files can be given at once, in which case ``-o`` names an output
directory.

How results are rendered
------------------------

The HTML reporter draws the parts of the page that every SET shares: header,
summary cards, AI summary and result grouping. Each individual result is drawn
by a *result renderer* chosen for the domain that produced it:

- ``languagemodel`` renders prompt / response (or conversation) results, and is
  the default when no other renderer matches.
- ``continuallearning`` renders CL results, whose content is the findings of
  each evaluator rather than a prompt and a response. It also adds a
  *SET Configuration* section and, for multi-case reports, a
  *Results Overview* table.

Within a CL result, each evaluator's findings are drawn by a *detection
renderer* registered for that evaluator's name. Evaluators with no registered
renderer are shown in full by a generic fallback (verdict, reason and every
recorded value), so **a new CL SET produces a complete HTML report without any
changes to the report generation code**. To give an evaluator a purpose-built
view, register a detection renderer:

.. code-block:: python

   from avise.reportgen.reporters.html_renderers import (
       DetectionRenderer, register_detection_renderer,
   )
   from avise.reportgen.reporters.html_renderers.components import fmt_percent

   @register_detection_renderer
   class ForgettingRenderer(DetectionRenderer):
       evaluators = ("forgetting_rate",)   # key in the result's detections
       title = "Forgetting Rate"
       description = "Shown by the ? icon next to the title."
       glossary = {"rate": "Shown by the ? icon next to 'Rate' in the values table."}

       def headline(self, findings, context):
           # (label, value, tone, explanation) tiles shown under the verdict;
           # tone is "passed", "failed", "error" or "neutral".
           return [("Forgetting", fmt_percent(findings.get("rate")), "neutral",
                    "What the forgetting rate means.")]

       def summary_line(self, findings, context):
           return f"Rate {fmt_percent(findings.get('rate'))}"   # on the collapsed card

       def verdict(self, findings, susceptible, context):
           return None   # or {"rule", "conditions", "conclusion"} to explain the verdict

Each case opens with a *Why this case PASSED / FAILED* box. It is built from
the evaluators listed in the result's ``metadata["verdict_basis"]``, which a
SET sets in ``evaluate()`` to the evaluator(s) its status is derived from.
Configuration keys and task stages are explained through
``register_config_terms`` and ``register_stage_descriptions``. CL SETs record
each case's task sequence in the report configuration with
``BaseSETPipeline.describe_set_cases()``, which the *SET Configuration*
section shows as a stage table.

A new domain (for example a future SET family that is neither language model
nor continual learning) subclasses ``ResultRenderer`` and registers it with
``@register_result_renderer``. A SET can also select a renderer explicitly by
setting ``metadata["report_renderer"]`` on its results to the renderer's name.
To regenerate that domain's HTML from JSON, register its schema classes with
``avise.reportgen.loader.register_report_schema``.

AI summary
----------

When enabled, the AI summary is written by a local language model from the
same result data that is saved to the JSON report. What the model is given
depends on the SET's domain and comes from a *summary profile*
(``avise.reportgen.summarizers.profiles``):

- ``languagemodel`` lists the failed cases and their reasons (and is the
  default).
- ``continuallearning`` gives the model the attack configuration, each case's
  task sequence, the verdict explanation and every evaluator's interpreted
  result, using the same interpretation as the HTML report, so the summary
  cannot contradict the report about what a measurement means.

A CL SET can register a plain-language description and a list of mitigations
the model may choose from:

.. code-block:: python

   from avise.reportgen.summarizers.profiles import (
       CLSetSummaryInfo, register_cl_set_summary,
   )

   register_cl_set_summary("MySET", CLSetSummaryInfo(
       description="What the SET tests, in plain words.",
       remediation_options=["A mitigation (applies when ...)."],
   ))

A new domain subclasses ``SummaryProfile`` and registers it with
``@register_summary_profile``. The notes under the summary are produced by the
profile without the model.
