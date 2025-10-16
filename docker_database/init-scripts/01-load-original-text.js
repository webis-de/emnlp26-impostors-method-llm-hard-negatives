// This script runs automatically inside the MongoDB container
// when the database is first initialized (empty /data/db)

print(" Starting initialization: loading original_text dataset...");

// Select the database defined in environment of YAML file (or default to "original_text")
const dbName = _getEnv("MONGO_INITDB_DATABASE") || "original_text";
db = db.getSiblingDB(dbName);   // Returns database object

// Define collection name
const collectionName = "original_text"; 


// collection creation
if (!db.getCollectionNames().includes(collectionName)) {
  db.createCollection(collectionName);
  print(`Created collection: ${collectionName}`);
} else {
  print(`Collection '${collectionName}' already exists. Skipping creation.`);
}

// unique fields
db[collectionName].createIndex({ text_id: 1 }, { unique: true });   // TODO: do i need this? mongo will create _id automatically
db[collectionName].createIndex({ text_hash: 1 }, { unique: true });

// dataset base path from container environment variable
const datasetBasePath = _getEnv("DATASET_PATH") + "/student_essays/Intro2006/";
const datasetName = "student_essays";
print(`${datasetName} dataset base path: ${datasetBasePath}`);

// counter for text_id
let lastDoc = db[collectionName].find().sort({ text_id: -1 }).limit(1).toArray();
let textIdCounter = (lastDoc.length > 0) ? lastDoc[0].text_id + 1 : 0;
const docsToInsert = [];

// list directories under datasetBasePath
const allDirs = ls(datasetBasePath);

for (let i = 0; i < allDirs.length; i++) {
  const dirName = allDirs[i];

  // only process directories beginning with "Ass"
  if (!dirName.startsWith("Ass")) continue;

  const assignmentDir = datasetBasePath + dirName + "/";
  print(`Processing assignment directory: ${dirName}`);

  // list txt files in this assignment directory
  const files = ls(assignmentDir);
  for (let j = 0; j < files.length; j++) {
    const fileName = files[j];

    // only process .txt files
    if (!fileName.endsWith(".txt")) continue;

    const filePath = assignmentDir + fileName;
    let textContent;
    try {
      textContent = cat(filePath);
    } catch (e) {
      print(`Could not read file: ${filePath}`, e);
      continue;
    }
    let authorName = fileName.replace(".txt", "");

    // Simple MD5 hash function for mongosh (init script)
    function md5(str) {
        const crypto = require('crypto');
        return crypto.createHash('md5').update(str).digest('hex');
    }
    const textHash = md5(textContent);  // not reassigned, unique per text

    // check if document already exists
    const exists = db[collectionName].findOne({ author: authorName, assignment: dirName, text_hash: textHash });
    if (exists) continue;

    // create document
    const doc = {
      text_id: textIdCounter++,
      text: textContent,
      author: authorName,
      dataset: datasetName,
      assignment: dirName,
      text_hash: textHash
    };

    docsToInsert.push(doc);
  }
}

// insert documents
if (docsToInsert.length > 0) {
    try {
        const result = db[collectionName].insertMany(docsToInsert, { ordered: false });
        print(`Inserted ${result.insertedIds.length} documents into '${collectionName}'`);
    } catch (e) {
        if (e.code === 11000) {
            print("Duplicate key error encountered during insertMany. Some documents may already exist.");
        } else {
            throw e; // rethrow if it's a different error
        }
    }
} else {
  print("No documents found to insert.");
}

print("Initialization complete.");