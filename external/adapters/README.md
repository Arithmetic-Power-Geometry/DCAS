# Adapter contract

An adapter must run the *upstream* evaluator/author algorithm without changing its numerical logic and must emit the common result schema.  The DCAS adapter must use the frozen public DCAS core.  Each adapter records the upstream git commit and evaluator identifier.  If a source is unavailable or its documented command cannot be verified, the adapter must stop rather than substitute a reimplementation.
