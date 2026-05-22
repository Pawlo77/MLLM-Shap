# Metareview
This demo paper provides an open source platform for computing shapely values (SV) for multi-modal models, covering both text and audio. All the reviewers are supportive of the contribution and significance of this work and have positive things to say about the engineering quality.

**Pros:**
* All reviewers like the presentation of the work and agree with the practical significance of extending xAI to text-audio models.
* The underlying engineering is also agreed to be high quality, praising specific choices such as the modular design of the system and coalition-space reduction from the SGPA implementation.
* Most of the reviewers agree on the ease of reproducibility of the work which makes it a good contribution for the community

**Notes to improve:**
* Most reviewers highlight that testing is limited to one model (which the authors already acknowledge), so it would be great if the authors can expand on this in future versions.
* Reviewers mcwu and WjZf have some valid implementation concerns such as need for local GPUs, docker, and extending to custom estimator or utility functions for which additional guidance can be provided in the code and/or paper.
* The authors should incorporate these changes and other suggestions from the reviewers in the camera-ready version.
