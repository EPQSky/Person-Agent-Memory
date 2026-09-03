(function (scope) {
  'use strict';

  function capture(libraryId, document, commit) {
    const selection = {
      libraryId,
      path: document.path,
      commit,
      expectedSourceVersion: document.source_version,
    };
    if (document.token !== undefined) selection.contextToken = document.token;
    return Object.freeze(selection);
  }

  function request(selection, operationId) {
    return Object.freeze({
      url: `/api/v1/libraries/${selection.libraryId}/history/restore`,
      body: Object.freeze({
        path: selection.path,
        commit: selection.commit,
        expected_source_version: selection.expectedSourceVersion,
        operation_id: operationId,
        actor_type: 'user',
        source: 'web-history',
      }),
    });
  }

  function selectionStore() {
    let selected = null;
    return Object.freeze({
      select(libraryId, document, commit) {
        selected = capture(libraryId, document, commit);
        return selected;
      },
      clear() {
        selected = null;
      },
      current() {
        return selected;
      },
    });
  }

  function contextStore() {
    let sequence = 0;
    let current = null;

    function freeze(libraryId, path, sourceVersion, content) {
      return Object.freeze({
        token: ++sequence,
        libraryId,
        path,
        sourceVersion,
        content,
      });
    }

    function selectLibrary(libraryId) {
      current = freeze(libraryId, null, null, null);
      return current;
    }

    function requestDocument(libraryId, path) {
      current = freeze(libraryId, path, null, null);
      return current;
    }

    function acceptDocument(requestContext, document) {
      if (!isCurrent(requestContext)) return null;
      current = Object.freeze({
        token: requestContext.token,
        libraryId: requestContext.libraryId,
        path: document.path,
        sourceVersion: document.source_version,
        content: document.content,
      });
      return current;
    }

    function edit(content) {
      if (!current || !current.path || !current.sourceVersion) return null;
      current = freeze(current.libraryId, current.path, current.sourceVersion, content);
      return current;
    }

    function snapshot() {
      return current;
    }

    function isCurrent(context) {
      return Boolean(current && context && current.token === context.token);
    }

    return Object.freeze({
      selectLibrary,
      requestDocument,
      acceptDocument,
      edit,
      snapshot,
      isCurrent,
    });
  }

  function previewRequest(context) {
    return Object.freeze({
      url: `/api/v1/libraries/${context.libraryId}/document/preview`,
      body: Object.freeze({
        path: context.path,
        content: context.content,
        expected_source_version: context.sourceVersion,
      }),
    });
  }

  function saveRequest(context, operationId) {
    return Object.freeze({
      url: `/api/v1/libraries/${context.libraryId}/document`,
      body: Object.freeze({
        path: context.path,
        content: context.content,
        expected_source_version: context.sourceVersion,
        operation_id: operationId,
        actor_type: 'user',
        source: 'web',
      }),
    });
  }

  const api = Object.freeze({
    capture,
    request,
    selectionStore,
    contextStore,
    previewRequest,
    saveRequest,
  });
  scope.PamHistory = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window === 'undefined' ? globalThis : window);
