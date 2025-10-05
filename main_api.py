import uvicorn
from ms_graphrag_neo4j.api import app
import os
if __name__ == "__main__":
    # Use the port Render provides, default to 5468 if running locally
    port = int(os.environ.get("PORT", 5468))
    uvicorn.run(app, host="0.0.0.0", port=port)